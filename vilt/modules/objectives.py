import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import logging

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from vilt.modules.dist_utils import all_gather


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance

def compute_seq2seq(pl_module, batch):
    if pl_module.training:
        infer = pl_module.infer(batch, mask_text=True, mask_image=False,
                                seq2seq=True)
        #with torch.cuda.amp.autocast(enabled=pl_module.hparams.config['head_amp']):
        seq2seq_logits = pl_module.seq2seq_score(infer["text_feats"])
        seq2seq_labels = infer["text_labels"]

        logits = seq2seq_logits.view(-1, pl_module.hparams.config["vocab_size"])
        target = seq2seq_labels.view(-1)

        valid = target != -100
        logits = logits[valid]
        target = target[valid]

        logits = logits.float()
        if len(target) == 0:
            seq2seq_loss = torch.tensor(0., device=logits.device, requires_grad=True)
        else:
            seq2seq_loss = pl_module.seq_loss(logits, target)

        ret = {
            "seq_loss": seq2seq_loss,
            "seq_logits": seq2seq_logits,
            "seq_labels": seq2seq_labels,
            "seq_ids": infer["text_ids"],
        }
        if pl_module.hparams.config.get('ema_distill_seq'):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    ma_infer = pl_module.infer(batch, mask_text=True, mask_image=False,
                                            seq2seq=True)
                    ma_seq2seq_logits = pl_module.seq2seq_score(ma_infer["text_feats"])
                    ma_logits = ma_seq2seq_logits.view(-1, pl_module.hparams.config["vocab_size"])
                    ma_logits = ma_logits[valid]
                    ma_logits = ma_logits.float()
            ma_seq2seq_loss = torch.nn.functional.kl_div(
                logits.log_softmax(dim=1),
                ma_logits.log_softmax(dim=1),
                reduction='batchmean',
                log_target=True,
            )
            weight = pl_module.hparams.config.get('ema_distill_seq_weight', 1)
            ret['ma_seq_loss'] = ma_seq2seq_loss * weight
    else:
        # not ready
        image_embeds, image_masks = pl_module.encode_image(batch['img'][0])
        ret = pl_module.generate(
            img_feats=image_embeds,
            **pl_module.caption_test_extra_input,
        )

    return ret

def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)

    #with torch.cuda.amp.autocast(enabled=pl_module.hparams.config['head_amp']):
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    logits = mlm_logits.view(-1, pl_module.hparams.config["vocab_size"])
    target = mlm_labels.view(-1)
    valid = target != -100
    logits = logits[valid]
    target = target[valid]

    logits = logits.float()
    if len(target) == 0:
        mlm_loss = torch.tensor(0., device=logits.device, requires_grad=True)
    else:
        mlm_loss = pl_module.mlm_loss(logits, target)

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    if pl_module.hparams.config.get('ema_distill_mlm'):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                ma_infer = pl_module.ma_teacher.infer(batch, mask_text=True, mask_image=False)
                ma_mlm_logits = pl_module.ma_teacher.mlm_score(ma_infer["text_feats"])
                ma_logits = ma_mlm_logits.view(-1, pl_module.hparams.config["vocab_size"])
                ma_logits = ma_logits[valid]
                ma_logits = ma_logits.float()
        ma_mlm_loss = torch.nn.functional.kl_div(
            logits.log_softmax(dim=1),
            ma_logits.log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        weight = pl_module.hparams.config.get('ema_distill_mlm_weight') or 1.
        ret['ma_mlm_loss'] = ma_mlm_loss * weight

    return ret

def compute_decoder(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    ret = infer

    if pl_module.hparams.config.get('ema_distill_mlm'):
        raise NotImplementedError('not ready')
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                ma_infer = pl_module.ma_teacher.infer(batch, mask_text=True, mask_image=False)
                ma_mlm_logits = pl_module.ma_teacher.mlm_score(ma_infer["text_feats"])
                ma_logits = ma_mlm_logits.view(-1, pl_module.hparams.config["vocab_size"])
                ma_logits = ma_logits[valid]
                ma_logits = ma_logits.float()
        ma_mlm_loss = torch.nn.functional.kl_div(
            logits.log_softmax(dim=1),
            ma_logits.log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        weight = pl_module.hparams.config.get('ema_distill_mlm_weight') or 1.
        ret['ma_mlm_loss'] = ma_mlm_loss * weight

    return ret

def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpp_logits = pl_module.mpp_score(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mppd(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mppd_logits = pl_module.mppd_score(infer["image_feats"])
    mppd_labels = infer["image_labels_mppd"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mppd_labels[filter_to_train]
    logits = mppd_logits[filter_to_train]
    mppd_loss = F.mse_loss(logits, labels)

    ret = {
        "mppd_loss": mppd_loss,
        "mppd_logits": mppd_logits,
        "mppd_labels": mppd_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mppd_loss")(ret["mppd_loss"])
    pl_module.log(f"mppd/{phase}/loss", loss)

    return ret


def compute_mpfr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpfr_logits = pl_module.mpfr_score(infer["image_feats"])
    mpfr_labels = infer["image_labels_mpfr"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mpfr_labels[filter_to_train]
    logits = mpfr_logits[filter_to_train]
    mpfr_loss = F.mse_loss(logits, labels)

    ret = {
        "mpfr_loss": mpfr_loss,
        "mpfr_logits": mpfr_logits,
        "mpfr_labels": mpfr_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpfr_loss")(ret["mpfr_loss"])
    pl_module.log(f"mpfr/{phase}/loss", loss)

    return ret


def compute_itm_wpa(pl_module, batch):
    if pl_module.training:
        return compute_itm_wpa_train(pl_module, batch)
    else:
        return compute_itm_wpa_test(pl_module, batch)

def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    batch = {k: v for k, v in batch.items()}
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
    num_res_factors = len(pl_module.hparams.config['multi_res_factors'])
    if 'false_image_0' not in batch:
        if num_res_factors == 0:
            rand_idx = torch.randperm(len(batch['image'][0]))
            batch['false_image_0'] = [batch['image'][0][rand_idx]]
            batch['correct_false'] = (batch['idx_img'] != batch['idx_img'][rand_idx]).long()
        else:
            num_image = len(batch['image'][0])
            shuffle_len = num_image // 2 + 1
            idx = torch.cat((torch.randperm(shuffle_len), torch.arange(shuffle_len, num_image)))
            batch['image'] = [batch['image'][0][idx]]
            for i in range(num_res_factors):
                k = 'image_{}'.format(i)
                batch[k] = [batch[k][0][idx]]
            batch['correct_false'] = (batch['idx_img'] != batch['idx_img'][idx]).long()
    else:
        assert num_res_factors == 0

    if 'correct_false' in batch:
        # we need to overwrite itm_labels
        # (itm_label, correct_false): (0, 1) -> 1 and -> 0 otherwise
        itm_labels = 1 - (1 - itm_labels) * batch['correct_false']

    if num_res_factors == 0:
        itm_images = [
            torch.stack(
                [
                    ti if itm_labels[i] == 1 else fi
                    for i, (ti, fi) in enumerate(zip(bti, bfi))
                ]
            )
            for bti, bfi in zip(batch["image"], batch["false_image_0"])
        ]
        batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    #with torch.cuda.amp.autocast(enabled=pl_module.hparams.config['head_amp']):
    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_logits = itm_logits.float()
    #itm_loss = F.cross_entropy(itm_logits, itm_labels.long())
    itm_loss = pl_module.itm_loss(itm_logits, itm_labels.long())
    ret = {
        'itm_loss': itm_loss,
        'itm_logits': itm_logits,
    }

    if pl_module.ma_teacher and pl_module.hparams.config.get('ema_distill_itm'):
        with torch.no_grad():
            ma_infer = pl_module.ma_teacher.infer(batch, mask_text=False, mask_image=False)
            ma_itm_logits = pl_module.ma_teacher.itm_score(ma_infer["cls_feats"])
            ma_itm_logits = ma_itm_logits.float()
        ma_itm_loss = torch.nn.functional.kl_div(
            itm_logits.log_softmax(dim=1),
            ma_itm_logits.log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        weight = pl_module.hparams.config.get('ma_itm_loss_weight', 1)
        ret['ma_itm_loss'] = ma_itm_loss * weight

    return ret

def compute_itm_wpa_test(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    batch = {k: v for k, v in batch.items()}
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    for k in ['text_feats', 'image_feats']:
        infer[k] = infer[k].float()

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_logits = itm_logits.float()
    #itm_loss = F.cross_entropy(itm_logits, itm_labels.long())
    #itm_loss = pl_module.itm_loss(itm_logits, itm_labels.long())

    ret = {
        #"itm_loss": itm_loss,
        "itm_logits": itm_logits,
        #"itm_labels": itm_labels,
    }

    return ret

def compute_itm_wpa_train(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    batch = {k: v for k, v in batch.items()}
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
    num_res_factors = len(pl_module.hparams.config['multi_res_factors'])
    if 'false_image_0' not in batch:
        if num_res_factors == 0:
            rand_idx = torch.randperm(len(batch['image'][0]))
            batch['false_image_0'] = [batch['image'][0][rand_idx]]
            batch['correct_false'] = (batch['idx_img'] != batch['idx_img'][rand_idx]).long()
        else:
            num_image = len(batch['image'][0])
            shuffle_len = num_image // 2 + 1
            idx = torch.cat((torch.randperm(shuffle_len), torch.arange(shuffle_len, num_image)))
            batch['image'] = [batch['image'][0][idx]]
            for i in range(num_res_factors):
                k = 'image_{}'.format(i)
                batch[k] = [batch[k][0][idx]]
            batch['correct_false'] = (batch['idx_img'] != batch['idx_img'][idx]).long()
    else:
        assert num_res_factors == 0

    if 'correct_false' in batch:
        # we need to overwrite itm_labels
        # (itm_label, correct_false): (0, 1) -> 1 and -> 0 otherwise
        itm_labels = 1 - (1 - itm_labels) * batch['correct_false']

    if num_res_factors == 0:
        itm_images = [
            torch.stack(
                [
                    ti if itm_labels[i] == 1 else fi
                    for i, (ti, fi) in enumerate(zip(bti, bfi))
                ]
            )
            for bti, bfi in zip(batch["image"], batch["false_image_0"])
        ]
        batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    for k in ['text_feats', 'image_feats']:
        infer[k] = infer[k].float()

    #with torch.cuda.amp.autocast(enabled=pl_module.hparams.config['head_amp']):
    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_logits = itm_logits.float()
    #itm_loss = F.cross_entropy(itm_logits, itm_labels.long())
    itm_loss = pl_module.itm_loss(itm_logits, itm_labels.long())
    ret = {'itm_loss': itm_loss}

    if pl_module.hparams.config.get('ema_distill_itm'):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                ma_infer = pl_module.ma_teacher.infer(batch, mask_text=False, mask_image=False)
                for k in ['text_feats', 'image_feats']:
                    ma_infer[k] = ma_infer[k].float()
                ma_itm_logits = pl_module.ma_teacher.itm_score(ma_infer["cls_feats"])
                ma_itm_logits = ma_itm_logits.float()
        ma_itm_loss = torch.nn.functional.kl_div(
            itm_logits.log_softmax(dim=1),
            ma_itm_logits.log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        ema_weight = pl_module.hparams.config.get('ema_distill_itm_weight') or 1.
        ret['ma_itm_loss'] = ma_itm_loss * ema_weight

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["image_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = trace(cost.matmul(T.detach()))

    if pl_module.hparams.config.get('balance_wpa'):
        from qd.layers.loss import get_balanced_weight
        weight = get_balanced_weight(
            distance, alpha=pl_module.hparams.config.get('equal_sce_alpha', 0.5))
        distance = weight * distance

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    wpa_loss_weight = pl_module.hparams.config.get('wpa_loss_weight', 0.1)
    ret["itm_wpa_loss"] = wpa_loss_weight * ot_loss

    return ret

def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(
        ret["imgcls_logits"], ret["imgcls_labels"]
    )
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret

class VQABinaryLoss(nn.Module):
    def forward(self, vqa_logits, vqa_targets):
        return F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1]  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

def compute_refcont(pl_module, batch):
    assert len(pl_module.hparams.config['multi_res_factors']) == 0
    img = batch['image'][0]
    feats1 = pl_module.extract_image_feature(
        img
    )
    with torch.no_grad():
        feats2 = pl_module.extract_image_feature(
            batch['image2'][0]
        )
    align_loss = pl_module.refcont_loss(feats1, feats2)
    loss_weight = pl_module.hparams.config['refcont_loss_weight']
    ret = {
        'refcont_loss': loss_weight * align_loss,
    }
    return ret

def compute_simclr(pl_module, batch):
    assert len(pl_module.hparams.config['multi_res_factors']) == 0
    img = batch['image'][0]
    feats1 = pl_module.extract_image_feature(
        img
    )
    feats2 = pl_module.extract_image_feature(
        batch['image2'][0]
    )
    align_loss = pl_module.simclr_loss(feats1, feats2)
    loss_weight = pl_module.hparams.config['simclr_loss_weight']
    ret = {
        'simclr_loss': loss_weight * align_loss,
    }
    return ret

def compute_ema_clip(pl_module, batch):
    infer = pl_module.infer(
        batch,
        mask_text=False,
        mask_image=False,
        sep_process=True,
    )
    with torch.no_grad():
        # in apex/o2, ma_teacher should be fp32 and we can run it with amp;
        # in pytorch/amp or fairscale, enable it is also fine
         with torch.cuda.amp.autocast(enabled=True):
             ma_infer = pl_module.ma_teacher.infer(
                 batch,
                 mask_text=False,
                 mask_image=False,
                 sep_process=True,
             )
    if pl_module.hparams.config.get('ema_distill_clip'):
        # we need to first calculate this and ignore to update the queue. so
        # that the negative queue is the same
        ma_align_loss = pl_module.clip_align_loss(
            {
                'img_feats': ma_infer['image_feats'],
                'idx_img': batch['idx_img'],
            },
            {
                'text_feats': ma_infer['text_feats'],
                'input_ids': batch['text_ids'],
                'origin_input_ids': batch['text_ids'],
            },
            ema_image={
                'img_feats': ma_infer['image_feats'],
            },
            ema_text={
                'text_feats': ma_infer['text_feats'],
                'input_ids': batch['text_ids'],
                'origin_input_ids': batch['text_ids'],
            },
            return_info=True,
            ignore_update_queue=True,
        )
    align_loss = pl_module.clip_align_loss(
        {
            'img_feats': infer['image_feats'],
            'idx_img': batch['idx_img'],
        },
        {
            'text_feats': infer['text_feats'],
            'input_ids': batch['text_ids'],
            'origin_input_ids': batch['text_ids'],
        },
        ema_image={
            'img_feats': ma_infer['image_feats'],
        },
        ema_text={
            'text_feats': ma_infer['text_feats'],
            'input_ids': batch['text_ids'],
            'origin_input_ids': batch['text_ids'],
        },
        return_info=True,
    )
    clip_loss_weight = pl_module.hparams.config['clip_loss_weight']
    ret = {}
    ret['eclip_loss'] = clip_loss_weight * align_loss['loss']
    if pl_module.hparams.config.get('ema_distill_clip'):
        ma_image_loss = torch.nn.functional.kl_div(
            align_loss['image_logits'].log_softmax(dim=1),
            ma_align_loss['image_logits'].log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        ma_text_loss = torch.nn.functional.kl_div(
            align_loss['text_logits'].log_softmax(dim=1),
            ma_align_loss['text_logits'].log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        weight = pl_module.hparams.config.get('ema_distill_clip_weight') or 1.
        ret['ma_eclip_loss'] = (ma_image_loss + ma_text_loss) / 2. * weight

    return ret

def compute_fork_ema_clip(pl_module, batch):
    infer = pl_module.infer(
        batch,
        mask_text=False,
        mask_image=False,
        no_fuse=True,
    )
    with torch.no_grad():
        # in apex/o2, it should be no harm to wrap it with amp
        with torch.cuda.amp.autocast(enabled=True):
            ma_infer = pl_module.ma_teacher.infer(
                batch,
                mask_text=False,
                mask_image=False,
                no_fuse=True,
            )
    align_loss = pl_module.clip_align_loss(
        {
            'img_feats': infer['image_embeds'],
            'idx_img': batch['idx_img'],
        },
        {
            'text_feats': infer['text_embeds'],
            'input_ids': batch['text_ids'],
            'origin_input_ids': batch['text_ids'],
        },
        ema_image={
            'img_feats': ma_infer['image_embeds'],
        },
        ema_text={
            'text_feats': ma_infer['text_embeds'],
            'input_ids': batch['text_ids'],
            'origin_input_ids': batch['text_ids'],
        },
    )
    clip_loss_weight = pl_module.hparams.config['clip_loss_weight']
    ret = {
        'clip_align_loss': clip_loss_weight * align_loss,
    }

    return ret

def compute_fork_clip(pl_module, batch):
    infer = pl_module.infer(
        batch,
        mask_text=False,
        mask_image=False,
        no_fuse=True,
    )
    if pl_module.hparams.config['head_amp'] is None:
        align_loss_info = pl_module.clip_align_loss(
            {
                'img_feats': infer['image_embeds'],
                'idx_img': batch['idx_img'],
            },
            {
                'text_feats': infer['text_embeds'],
                'input_ids': batch['text_ids'],
                'origin_input_ids': batch['text_ids'],
            },
            return_info=True
        )
        align_loss = align_loss_info['loss']
        clip_loss_weight = pl_module.hparams.config['clip_loss_weight']
    else:
        with torch.cuda.amp.autocast(enabled=pl_module.hparams.config['head_amp']):
            align_loss_info = pl_module.clip_align_loss(
                {
                    'img_feats': infer['image_embeds'],
                    'idx_img': batch['idx_img'],
                },
                {
                    'text_feats': infer['text_embeds'],
                    'input_ids': batch['text_ids'],
                    'origin_input_ids': batch['text_ids'],
                },
                return_info=True
            )
            align_loss = align_loss_info['loss']
            clip_loss_weight = pl_module.hparams.config['clip_loss_weight']
    ret = {
        'clip_align_loss': clip_loss_weight * align_loss,
    }
    if pl_module.ma_teacher and pl_module.hparams.config.get('ma_teacher_clip'):
        # not tested here
        with torch.no_grad():
            ma_infer = pl_module.ma_teacher.infer(
                batch,
                mask_text=False,
                mask_image=False,
                no_fuse=True,
            )
            ma_align_loss_info = pl_module.ma_teacher.clip_align_loss(
                {
                    'img_feats': ma_infer['image_feats'],
                    'idx_img': batch['idx_img'],
                },
                {
                    'text_feats': ma_infer['text_feats'],
                    'input_ids': batch['text_ids'],
                    'origin_input_ids': batch['text_ids'],
                },
                return_info=True
            )
        ma_clip_loss1 = torch.nn.functional.kl_div(
            ma_align_loss_info['logits'].log_softmax(dim=1),
            align_loss_info['logits'].log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        ma_clip_loss2 = torch.nn.functional.kl_div(
            ma_align_loss_info['logits'].t().log_softmax(dim=1),
            align_loss_info['logits'].t().log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        from qd.qd_common import get_mpi_size
        ret['ma_clip_loss'] = get_mpi_size() * clip_loss_weight * (ma_clip_loss1 + ma_clip_loss2) / 2.

    return ret

def compute_clip(pl_module, batch):
    infer = pl_module.infer(
        batch,
        mask_text=False,
        mask_image=False,
        sep_process=True,
    )
    #with torch.cuda.amp.autocast(enabled=pl_module.hparams.config['head_amp']):
    align_loss_info = pl_module.clip_align_loss(
        {
            'img_feats': infer['image_feats'],
            'idx_img': batch['idx_img'],
        },
        {
            'text_feats': infer['text_feats'],
            'input_ids': batch['text_ids'],
            'origin_input_ids': batch['text_ids'],
        },
        return_info=True
    )
    align_loss = align_loss_info['loss']
    clip_loss_weight = pl_module.hparams.config['clip_loss_weight']
    ret = {
        'clip_align_loss': clip_loss_weight * align_loss,
    }
    if pl_module.ma_teacher and pl_module.hparams.config.get('ema_distill_clip'):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                ma_infer = pl_module.ma_teacher.infer(
                    batch,
                    mask_text=False,
                    mask_image=False,
                    sep_process=True,
                )
            ma_align_loss_info = pl_module.ma_teacher.clip_align_loss(
                {
                    'img_feats': ma_infer['image_feats'],
                    'idx_img': batch['idx_img'],
                },
                {
                    'text_feats': ma_infer['text_feats'],
                    'input_ids': batch['text_ids'],
                    'origin_input_ids': batch['text_ids'],
                },
                return_info=True
            )
        if pl_module.hparams.config.get('ema_distill_clip_correct_order'):
            ma_clip_loss1 = torch.nn.functional.kl_div(
                align_loss_info['logits'].log_softmax(dim=1),
                ma_align_loss_info['logits'].log_softmax(dim=1),
                reduction='batchmean',
                log_target=True,
            )
            ma_clip_loss2 = torch.nn.functional.kl_div(
                align_loss_info['logits'].t().log_softmax(dim=1),
                ma_align_loss_info['logits'].t().log_softmax(dim=1),
                reduction='batchmean',
                log_target=True,
            )
        else:
            ma_clip_loss1 = torch.nn.functional.kl_div(
                ma_align_loss_info['logits'].log_softmax(dim=1),
                align_loss_info['logits'].log_softmax(dim=1),
                reduction='batchmean',
                log_target=True,
            )
            ma_clip_loss2 = torch.nn.functional.kl_div(
                ma_align_loss_info['logits'].t().log_softmax(dim=1),
                align_loss_info['logits'].t().log_softmax(dim=1),
                reduction='batchmean',
                log_target=True,
            )
        from qd.qd_common import get_mpi_size
        w = pl_module.hparams.config.get('ema_distill_clip_weight') or 1.
        w *= clip_loss_weight
        ret['ma_clip_loss'] = get_mpi_size() * w * (ma_clip_loss1 + ma_clip_loss2) / 2.

    return ret

def compute_snli(pl_module, batch):
    ret = compute_snli_forward(pl_module, batch)
    if pl_module.ma_teacher:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                ma_ret = compute_snli_forward(pl_module.ma_teacher, batch)
        ma_loss = torch.nn.functional.kl_div(
            ret['snli_logits'].log_softmax(dim=1),
            ma_ret['snli_logits'].log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        weight = pl_module.hparams.config['ema_distill_weight']
        ret['ma_loss'] = ma_loss * weight
    return ret

def compute_snli_forward(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    snli_logits = pl_module.snli_classifier(infer["cls_feats"])

    snli_labels = batch["snli_labels"]
    snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()
    snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))

    ret = {
        "snli_loss": snli_loss,
        "snli_logits": snli_logits,
    }

    return ret

def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    if pl_module.hparams.config.get('vqa_use_last_token'):
        num = infer['text_masks'].shape[0]
        aug_input_ids = torch.cat((
            infer['text_masks'].int(), torch.zeros((num, 1), device=infer['text_masks'].device, dtype=torch.int)
        ), dim=1)
        idx = (aug_input_ids == 0).int().argmax(dim=1) - 1

        text_feats = infer['text_feats']
        cls_feats = text_feats[torch.arange(len(text_feats)), idx]
    else:
        cls_feats = infer["cls_feats"]
    vqa_logits = pl_module.vqa_classifier(cls_feats)
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(vqa_logits.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_logits = vqa_logits.float()
    vqa_loss = pl_module.vqa_loss(vqa_logits, vqa_targets)
    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    if pl_module.ma_teacher and pl_module.training:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                ma_infer = pl_module.ma_teacher.infer(batch, mask_text=False, mask_image=False)
            ma_vqa_logits = pl_module.ma_teacher.vqa_classifier(ma_infer["cls_feats"])
            ma_vqa_logits = ma_vqa_logits.float()
        ma_vqa_loss = F.binary_cross_entropy_with_logits(
            vqa_logits,
            vqa_targets.sigmoid(),
        ) * vqa_logits.shape[1]
        #ma_vqa_loss = torch.nn.functional.kl_div(
            #vqa_logits.log_softmax(dim=1),
            #ma_vqa_logits.log_softmax(dim=1),
            #reduction='batchmean',
            #log_target=True,
        #)
        weight = pl_module.hparams.config.get('ma_vqa_loss_weight', 1)
        ret['ma_vqa_loss'] = ma_vqa_loss * weight

    #phase = "train" if pl_module.training else "val"
    #loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    #score = getattr(pl_module, f"{phase}_vqa_score")(
        #ret["vqa_logits"], ret["vqa_targets"]
    #)
    #pl_module.log(f"vqa/{phase}/loss", loss)
    #pl_module.log(f"vqa/{phase}/score", score)

    return ret

def compute_nlvr2_forward(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1,
        image_key='image_left',
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2,
        image_key='image_right',
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answer"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    return ret

def compute_nlvr2(pl_module, batch):
    ret = compute_nlvr2_forward(pl_module, batch)
    if pl_module.ma_teacher:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                ma_ret = compute_nlvr2_forward(pl_module.ma_teacher, batch)
        ma_loss = torch.nn.functional.kl_div(
        ret['nlvr2_logits'].log_softmax(dim=1),
        ma_ret['nlvr2_logits'].log_softmax(dim=1),
            reduction='batchmean',
            log_target=True,
        )
        weight = pl_module.hparams.config['ema_distill_weight']
        ret['ma_loss'] = ma_loss * weight

    return ret

def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
