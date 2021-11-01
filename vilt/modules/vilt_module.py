import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
from transformers import RobertaConfig, RobertaModel, AlbertModel, DebertaModel, ElectraModel
from qd.layers.generation_by_mask import GenerationByMask


class ViLTransformerSS(pl.LightningModule, GenerationByMask):
    def __init__(self, config, image_encoder=None, text_encoder=None,
                 hidden2text_decoder=None
                 ):
        super().__init__()
        self.save_hyperparameters()

        if not self.hparams.config.get('disable_token_type_embedding'):
            self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
            self.token_type_embeddings.apply(objectives.init_weights)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        if self.image_encoder:
            self.head_transformer_image_transform = nn.Linear(config['clip_image_embed_size'], config['hidden_size'])
            self.head_transformer_image_transform.apply(objectives.init_weights)
        else:
            self.head_transformer_image_transform = None

        if self.text_encoder:
            self.head_transformer_text_transform = nn.Linear(config['clip_text_embed_size'], config['hidden_size'])
            self.head_transformer_text_transform.apply(objectives.init_weights)
        else:
            self.head_transformer_text_transform = None

        bert_config = self.get_bert_config()

        # gradually go to the path with old_order == False
        old_order = config['loss_names'].get('mlm', 0) > 0

        if (not self.text_encoder) and old_order:
            self.text_embeddings = BertEmbeddings(bert_config)
            self.text_embeddings.apply(objectives.init_weights)

        # this is actually the joint transformer
        self.transformer = self.build_transformer()

        if self.hparams.config.get('fuse_type') != 'cross':
            self.pooler = heads.Pooler(config["hidden_size"])
            self.pooler.apply(objectives.init_weights)

        if (not self.text_encoder) and (not old_order):
            self.text_embeddings = BertEmbeddings(bert_config)
            self.text_embeddings.apply(objectives.init_weights)

        self.build_loss_specific_module(config)

        self.hidden2text_decoder = hidden2text_decoder

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        if self.hparams.config['ma_teacher_momentum'] is not None:
            import copy
            self.ma_teacher = copy.deepcopy(self)
            from qd.torch_common import freeze_parameters
            freeze_parameters(self.ma_teacher)
        else:
            self.ma_teacher = None

        self.printed_loss_idx_info = False
        self.log_info = {}

    def get_bert_config(self):
        # do not
        config = self.hparams.config

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        return bert_config

    def build_loss_specific_module(self, config):
        bert_config = self.get_bert_config()

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)
            mlm_loss_type = config.get('mlm_loss_type')
            if config.get('mlm_label_smoothing') and not mlm_loss_type:
                mlm_loss_type = 'smooth'
            if not mlm_loss_type:
                self.mlm_loss = nn.CrossEntropyLoss()
            elif mlm_loss_type == 'smooth':
                from qd.layers.loss import SmoothLabelCrossEntropyLoss
                self.mlm_loss = SmoothLabelCrossEntropyLoss(log_prefix='mlm')
            elif mlm_loss_type == 'smooth_ignore_worst':
                from qd.layers.loss import SmoothLabelIgnoreWorstCrossEntropyLoss
                smooth_ignore_worst_ratio = config.get('smooth_ignore_worst_ratio', 2)
                self.mlm_loss = SmoothLabelIgnoreWorstCrossEntropyLoss(
                    worst_ratio=smooth_ignore_worst_ratio,
                    log_prefix='mlm',
                )
            elif mlm_loss_type == 'norm_ce':
                from qd.layers.loss import NormAndCrossEntropy
                t = config.get('norm_ce_temperature')
                self.mlm_loss = NormAndCrossEntropy(t, log_prefix='mlm')
            elif mlm_loss_type == 'equal_sce':
                from qd.layers.loss import EqualCrossEntropyLoss
                self.mlm_loss = EqualCrossEntropyLoss(
                    log_prefix='mlm',
                    alpha=config.get('equal_sce_alpha'),
                )
            else:
                raise NotImplementedError(mlm_loss_type)

        if config["loss_names"].get("seq2seq", 0) > 0:
            self.seq2seq_score = heads.MLMHead(bert_config)
            self.seq2seq_score.apply(objectives.init_weights)
            seq_loss_type = config.get('seq_loss_type')
            if not seq_loss_type:
                self.seq_loss = nn.CrossEntropyLoss()
            elif seq_loss_type == 'smooth':
                from qd.layers.loss import SmoothLabelCrossEntropyLoss
                self.seq_loss = SmoothLabelCrossEntropyLoss(log_prefix='seq')
            elif seq_loss_type == 'smooth_ignore_worst':
                from qd.layers.loss import SmoothLabelIgnoreWorstCrossEntropyLoss
                smooth_ignore_worst_ratio = config.get('smooth_ignore_worst_ratio', 2)
                self.seq_loss = SmoothLabelIgnoreWorstCrossEntropyLoss(
                    worst_ratio=smooth_ignore_worst_ratio, log_prefix='seq')
            elif seq_loss_type == 'norm_ce':
                from qd.layers.loss import NormAndCrossEntropy
                t = config.get('norm_ce_temperature')
                self.seq_loss = NormAndCrossEntropy(t, log_prefix='seq')
            elif seq_loss_type == 'equal_sce':
                from qd.layers.loss import EqualCrossEntropyLoss
                self.seq_loss = EqualCrossEntropyLoss(
                    log_prefix='seq',
                    alpha=config.get('equal_sce_alpha'),
                )
            else:
                raise NotImplementedError(seq_loss_type)

        if config["loss_names"]["itm"] > 0 or config["loss_names"].get("itm_only", 0) > 0:
            if self.hparams.config.get('fuse_type') == 'cross':
                # the concatenation of the text input and image input is the
                # module input
                self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
            else:
                self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)
            itm_loss_type = config.get('itm_loss_type')
            if not itm_loss_type:
                #self.itm_loss = nn.CrossEntropyLoss()
                from qd.layers.loss import CrossEntropyLossVerbose
                self.itm_loss = CrossEntropyLossVerbose(log_prefix='itm')
            elif itm_loss_type == 'smooth':
                from qd.layers.loss import SmoothLabelCrossEntropyLoss
                self.itm_loss = SmoothLabelCrossEntropyLoss()
            elif itm_loss_type == 'smooth_ignore_worst':
                from qd.layers.loss import SmoothLabelIgnoreWorstCrossEntropyLoss
                smooth_ignore_worst_ratio = config.get('smooth_ignore_worst_ratio', 2)
                self.itm_loss = SmoothLabelIgnoreWorstCrossEntropyLoss(
                    worst_ratio=smooth_ignore_worst_ratio,
                    log_prefix='itm',
                )
            elif itm_loss_type == 'equal_sce':
                from qd.layers.loss import EqualCrossEntropyLoss
                self.itm_loss = EqualCrossEntropyLoss(
                    log_prefix='itm',
                    label_smoothing=config.get('itm_label_smoothing', 0.1),
                    alpha=config.get('equal_sce_alpha'),
                )
            else:
                raise NotImplementedError(itm_loss_type)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            d = 2 if self.hparams.config.get('fuse_type') == 'cross' else 1
            if self.hparams.config.get('vqa_hidden_size'):
                vqa_hidden_size = self.hparams.config['vqa_hidden_size']
            else:
                vqa_hidden_size = 2 * hs
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * d, vqa_hidden_size),
                nn.LayerNorm(vqa_hidden_size),
                nn.GELU(),
                nn.Linear(vqa_hidden_size, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)
            if 'prior_prob' in config:
                prior_prob = config['prior_prob']
                from qd.torch_common import set_sigmoid_prob_prior_bias
                set_sigmoid_prob_prior_bias(self.vqa_classifier[-1].bias, prior_prob)
            loss_type = config.get('loss_type')
            if loss_type is None:
                # this is default choice
                self.vqa_loss = objectives.VQABinaryLoss()
            elif config['loss_type'] == 'bceByPos':
                from qd.layers.loss import BCELogitsNormByPositive
                self.vqa_loss = BCELogitsNormByPositive()
            elif loss_type == 'kl':
                self.vqa_loss = torch.nn.KLDivLoss(reduction="batchmean")
            else:
                raise NotImplementedError(loss_type)

        if self.hparams.config["loss_names"].get("snli", 0) > 0:
            d = 2 if self.hparams.config.get('fuse_type') == 'cross' else 1
            if self.hparams.config.get('snli_hidden_size'):
                snli_hidden_size = self.hparams.config['snli_hidden_size']
            else:
                snli_hidden_size = 2 * hs
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs * d, snli_hidden_size),
                nn.LayerNorm(snli_hidden_size),
                nn.GELU(),
                nn.Linear(snli_hidden_size, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        if self.hparams.config['loss_names'].get('clip', 0) > 0 or \
                self.hparams.config['loss_names'].get('fork_clip', 0) > 0:
            align_loss_type = self.hparams.config['clip_loss_type']
            temperature = self.hparams.config['clip_temperature']
            from qd.layers.image_text_align import create_align_loss
            self.clip_align_loss = create_align_loss(
                align_loss_type,
                temperature=temperature,
                loss_balanced=self.hparams.config.get('clip_balanced'),
                loss_balanced_alpha=self.hparams.config.get('equal_sce_alpha'),
                learnable_temperature=self.hparams.config.get('clip_learnable_temperature'),
            )

        if self.hparams.config['loss_names'].get('ema_clip', 0) > 0  or \
                self.hparams.config['loss_names'].get('fork_ema_clip', 0) > 0:
            align_loss_type = self.hparams.config['clip_loss_type']
            temperature = self.hparams.config['clip_temperature']
            from qd.layers.image_text_align import create_align_loss
            self.clip_align_loss = create_align_loss(
                align_loss_type,
                temperature=temperature,
                loss_balanced=self.hparams.config.get('clip_balanced'),
                loss_balanced_alpha=self.hparams.config.get('equal_sce_alpha'),
                loss_style='smoco',
                feat_dim=self.hparams.config['hidden_size'],
                queue_size=self.hparams.config['ema_clip_queue_size'],
                learnable_temperature=self.hparams.config.get('clip_learnable_temperature'),
            )

        if self.hparams.config['loss_names'].get('simclr', 0) > 0:
            from qd.layers.ntxent_loss import NTXentLoss
            temperature = self.hparams.config['simclr_temperature']
            self.simclr_loss = NTXentLoss(temperature, correct_loss=True)

        if self.hparams.config['loss_names'].get('refcont', 0) > 0:
            temperature = self.hparams.config['refcont_temperature']
            from qd.layers.loss import RefContLoss
            self.refcont_loss = RefContLoss(temperature)

    def build_transformer_default(self):
        if self.hparams.config["load_path"] == "":
            pretrain = self.hparams.config.get('pretrain', True)
        else:
            pretrain = False
        logging.info('pretrain = {}'.format(pretrain))
        if self.hparams.config['vit'].startswith('V'):
            # gradually to use this implementation to reduce the effort to
            # maintain two vision_transformer.py
            import timm
            import copy
            self.use_timm = True
            kwargs = copy.deepcopy(self.hparams.config)
            # teh two parameters are explicity specified
            for k in ['num_heads', 'patch_size']:
                if k in kwargs:
                    del kwargs[k]
            transformer = timm.create_model(
                model_name=self.hparams.config['vit'].lower(),
                pretrained=pretrain, **kwargs,
            )
        else:
            self.use_timm = False
            import vilt.modules.vision_transformer as vit
            transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=pretrain, config=self.hparams.config
            )

        if self.hparams.config.get('pipe_parallel'):
            blocks = transformer.blocks
            for b in blocks:
                def patch_block_forward(fwd):
                    def patched_forward(data):
                        x, mask = data[0], data[1]
                        x, attn, mask = fwd(x, mask)
                        return (x, mask)
                    return patched_forward
                b.forward = patch_block_forward(b.forward)
            transformer.blocks = nn.Sequential(*iter(blocks))

        return transformer

    def build_transformer_cross(self):
        bert_config = self.get_bert_config()
        config = self.hparams.config
        from qd.layers.zy_bert_model import BertCrossLayer

        self.head_transformer_image = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.head_transformer_image.apply(objectives.init_weights)

        self.head_transformer_text = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.head_transformer_text.apply(objectives.init_weights)

        self.head_transformer_image_pooler = heads.Pooler(config["hidden_size"])
        self.head_transformer_image_pooler.apply(objectives.init_weights)
        self.head_transformer_text_pooler = heads.Pooler(config["hidden_size"])
        self.head_transformer_text_pooler.apply(objectives.init_weights)

    def build_transformer_bert_ssa(self):
        bert_config = self.get_bert_config()
        config = self.hparams.config
        bert_config.num_hidden_layers = config['num_top_layer']

        from transformers.models.bert.modeling_bert import BertEncoder

        transformer = BertEncoder(bert_config)
        transformer.apply(objectives.init_weights)

        return transformer

    def build_transformer(self):
        fuse_type = self.hparams.config.get('fuse_type')
        if fuse_type is None:
            return self.build_transformer_default()
        elif fuse_type == 'cross':
            return self.build_transformer_cross()
        elif fuse_type == 'bert_ssa':
            return self.build_transformer_bert_ssa()

    def encode_text(self, text_ids, text_masks, seq2seq):
        if self.text_encoder is not None:
            if seq2seq:
                from qd.torch_common import construct_seq2seq_mask
                attention_mask = construct_seq2seq_mask(text_masks)
            else:
                attention_mask = text_masks
            # when retrun_dict = True, the returned object should be a special
            # dict, but fairscale.utils.containers.apply_to_tensors will maek
            # it as a dictionary, which does not support indexing, while teh
            # model inside still use the indexing, which will crash
            ret = self.text_encoder(
                input_ids=text_ids,
                attention_mask=attention_mask,
                return_dict=False)
            #return ret.last_hidden_state
            return ret[0]
        else:
            return self.text_embeddings(text_ids)

    def encode_image(self, img, ignore_cls=False, img_valid_mask=None):
        if self.image_encoder is not None:
            assert not ignore_cls, 'not implemented'
            image_embeds = self.image_encoder(img)
            image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=image_embeds.device)
            return image_embeds, image_masks
        else:
            sample_style = self.hparams.config.get('sample_style', 'vilt')
            if sample_style == 'vilt':
                (
                    image_embeds,
                    image_masks,
                    _, # patch_index,
                    _, # image_labels
                ) = self.transformer.visual_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=False,
                    ignore_cls=ignore_cls,
                )
            elif sample_style == 'rect_mask':
                image_embeds, image_masks = self.transformer.rect_patch_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    ignore_cls=ignore_cls,
                    img_valid_mask=img_valid_mask,
                )
            else:
                assert sample_style == 'rect'
                image_embeds = self.transformer.rect_patch_embed(
                    img, max_image_len=self.hparams.config["max_image_len"],
                    ignore_cls=ignore_cls,
                )
                image_masks = torch.ones(
                    image_embeds.shape[:2], dtype=torch.long,
                    device=image_embeds.device)
            return image_embeds, image_masks

    def process_one_modality(self, x, mask, num_text_tokens,
                             image_spatial_dims=None):
        if not self.use_timm:
            for i, blk in enumerate(self.transformer.blocks):
                x, mask = blk(x, mask=mask,
                                     num_text_tokens=num_text_tokens,
                                     spatial_dims=image_spatial_dims,
                                     )
        else:
            for i, blk in enumerate(self.transformer.blocks):
                x = blk(x, attention_mask=mask)
        x = self.transformer.norm(x)
        return x

    # used only in sim-clr. The code of encode_image_in_batch is incorrect
    def extract_image_feature(self, img):
        image_spatial_dims = img.shape[2:]
        image_embeds, image_masks = self.encode_image_in_batch(img)
        if not self.hparams.config.get('disable_token_type_embedding'):
            image_embeds = image_embeds + self.token_type_embeddings(
                    torch.full_like(image_masks, 1)
                )
        image_feats = self.process_one_modality(image_embeds, image_masks, 0,
                                                image_spatial_dims)
        image_feats = image_feats[:, 0]
        # if we want to disable normalization here, we can add a parameter in
        # this function
        image_feats = F.normalize(image_feats, dim=1)
        return image_feats

    def encode_image_in_batch(self, batch, key='image'):
        num_extra_res = len(self.hparams.config['multi_res_factors'])
        image_keys = ['{}_{}'.format(key, i) for i in range(num_extra_res)]
        img = batch[key][0]
        image_embeds, image_masks = self.encode_image(
            img, img_valid_mask=batch.get('{}_valid_mask'.format(key), [None])[0])
        if len(image_keys) > 0:
            assert image_keys[0] + '_vaid_mask' not in batch, 'not implemented'
            imbed_masks = [self.encode_image(batch[k][0], ignore_cls=True) for k in image_keys]
            image_embeds = torch.cat([image_embeds] + [e for e, _ in imbed_masks], dim=1)
            image_masks = torch.cat([image_masks] + [m for _, m in imbed_masks], dim=1)
        return image_embeds, image_masks

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        sep_process=False,
        seq2seq=False,
        no_fuse=False,
        image_key='image',
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch.get(f"text_labels{do_mlm}")
        text_masks = batch["text_masks"]
        ret = {}
        text_embeds = self.encode_text(text_ids, text_masks, seq2seq)

        if self.head_transformer_text_transform:
            text_embeds = self.head_transformer_text_transform(text_embeds)

        if image_embeds is None and image_masks is None:
            assert not mask_image, 'not supported'
            if 'image' in batch:
                image_spatial_dims = batch['image'][0].shape[2:]
            else:
                image_spatial_dims = None
            image_embeds, image_masks = self.encode_image_in_batch(
                batch, key=image_key)
            if self.head_transformer_image_transform:
                image_embeds = self.head_transformer_image_transform(image_embeds)

        if no_fuse:
            # these two could be used for contrastive loss. We do not use the
            # embeds after adding token type embedding because that token type
            # embeding is designed for fusion
            ret['image_embeds'] = image_embeds
            ret['text_embeds'] = text_embeds
            return ret

        if not self.hparams.config.get('disable_token_type_embedding'):
            device = self.token_type_embeddings.weight.device
            text_embeds, image_embeds = (
                text_embeds + self.token_type_embeddings(
                    torch.zeros_like(text_masks, device=device)),
                image_embeds
                + self.token_type_embeddings(
                    torch.full_like(image_masks, image_token_type_idx,
                                    device=device)
                ),
            )

        if not sep_process:
            text_feats, image_feats, cls_feats = self.fuse_modalities(
                text_embeds, text_masks,
                image_embeds, image_masks,
                image_spatial_dims,
                seq2seq
            )
        else:
            assert not seq2seq
            text_feats = self.process_one_modality(text_embeds, text_masks, text_embeds.shape[1],
                                 image_spatial_dims)
            image_feats = self.process_one_modality(image_embeds, image_masks, 0,
                                  image_spatial_dims)
            cls_feats = None

        ret.update({
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            #"raw_cls_feats": x[:, 0],
            #"image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            #"patch_index": patch_index,
        })

        if self.hidden2text_decoder:
            assert not sep_process, 'loggically not reasonable'
            hidden_states = torch.cat((image_feats, text_feats), dim=1)
            hidden_valid_mask = torch.cat((image_masks, text_masks), dim=1).bool()
            update = self.hidden2text_decoder(
                hidden_states,
                hidden_valid_mask=hidden_valid_mask,
                input_ids=batch.get('decoder_text_ids'),
                input_valid_mask=batch.get('decoder_text_masks'),
            )
            ret.update(update)

        return ret

    def fuse_modalities(self, text_embeds, text_masks,
                        image_embeds, image_masks,
                        image_spatial_dims,
                        seq2seq):
        fuse_type = self.hparams.config.get('fuse_type')
        if fuse_type is None:
            return self.fuse_modalities_by_self_attn(text_embeds, text_masks,
                                                     image_embeds, image_masks,
                                                     image_spatial_dims,
                                                     seq2seq,
                                                     )
        elif fuse_type == 'cross':
            return self.fuse_modalities_by_cross(text_embeds, text_masks,
                                                 image_embeds, image_masks,
                                                 image_spatial_dims,
                                                 seq2seq,
                                                 )
        elif fuse_type == 'bert_ssa':
            return self.fuse_modalities_by_bert_ssa(text_embeds, text_masks,
                                                     image_embeds, image_masks,
                                                     image_spatial_dims,
                                                     seq2seq,
                                                     )

    def fuse_modalities_by_bert_ssa(self, text_embeds, text_masks,
                        image_embeds, image_masks,
                        image_spatial_dims,
                        seq2seq):
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        # valid = 1 in masks
        if not seq2seq:
            co_masks = torch.cat([text_masks, image_masks], dim=1)
        else:
            from qd.torch_common import construct_seq2seq_mask
            co_masks = construct_seq2seq_mask(text_masks, image_masks)
        x = co_embeds

        from qd.torch_common import get_extended_attention_mask
        extended_mask = get_extended_attention_mask(
            attention_mask=co_masks,
            input_shape=co_masks.shape[:2],
            is_decoder=False,
        ).to(text_embeds.dtype)

        x = self.transformer(x, attention_mask=extended_mask)
        x = x.last_hidden_state
        #for i, blk in enumerate(self.transformer):
            #x, _attn, co_masks = blk(x, mask=co_masks)
        if not seq2seq:
            # itm loss needs the image masks. the iamge masks may be
            # updated by down-samplying in the transformers
            image_masks = co_masks[:, text_masks.shape[1]:]
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)
        return text_feats, image_feats, cls_feats

    def fuse_modalities_by_self_attn(self, text_embeds, text_masks,
                        image_embeds, image_masks,
                        image_spatial_dims,
                        seq2seq):
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        # valid = 1 in masks
        if not seq2seq:
            co_masks = torch.cat([text_masks, image_masks], dim=1)
        else:
            from qd.torch_common import construct_seq2seq_mask
            co_masks = construct_seq2seq_mask(text_masks, image_masks)
        x = co_embeds

        if self.hparams.config.get('pipe_parallel'):
            data = (x, co_masks)
            origin_device = x.device
            data = self.transformer.blocks(data)
            x, co_masks = data[0].to(origin_device), data[1].to(origin_device)
        else:
            for i, blk in enumerate(self.transformer.blocks):
                x, co_masks = blk(x, mask=co_masks,
                                  num_text_tokens=text_embeds.shape[1],
                                  spatial_dims=image_spatial_dims,
                                  )
        if not seq2seq:
            # itm loss needs the image masks. the iamge masks may be
            # updated by down-samplying in the transformers
            image_masks = co_masks[:, text_masks.shape[1]:]
        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)
        return text_feats, image_feats, cls_feats

    def fuse_modalities_by_cross(self, text_embeds, text_masks,
                        image_embeds, image_masks,
                        image_spatial_dims,
                        seq2seq):
        assert not seq2seq, 'not supported'

        input_shape = text_masks.size()
        from qd.torch_common import get_extended_attention_mask
        extend_text_masks = get_extended_attention_mask(
            text_masks, input_shape, is_decoder=False,
        ).to(text_embeds.dtype)

        input_shape = image_masks.size()
        extend_image_masks = get_extended_attention_mask(
            image_masks, input_shape, is_decoder=False,
        ).to(image_embeds.dtype)

        x, y = text_embeds, image_embeds

        for text_layer, image_layer in zip(self.head_transformer_text, self.head_transformer_image):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.head_transformer_text_pooler(x)
        cls_feats_image = self.head_transformer_image_pooler(y)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        return text_feats, image_feats, cls_feats

    def update_ma_teacher(self):
        ma = self.hparams.config['ma_teacher_momentum']
        with torch.no_grad():
            name2param = dict(self.named_parameters())
            for n, ma_param in self.ma_teacher.named_parameters():
                param = name2param[n]
                ma_param.data = ma_param.data * ma + param.data * (1. - ma)

    def reset_loss_index(self):
        self.loss_index = 0

    def forward(self, batch):
        if '_fw_bk_idx' in batch and \
                '_fw_bk_num' in batch:
            if batch['_fw_bk_num'] == len(self.current_tasks):
                if not self.printed_loss_idx_info:
                    logging.info('selecting one task from {}'.format(self.current_tasks))
                    self.printed_loss_idx_info = True
                self.current_tasks = [self.current_tasks[batch['_fw_bk_idx']]]
                not_update_ma_teacher = batch['_fw_bk_idx'] != 0
            else:
                assert batch['_fw_bk_num'] == 1 and batch['_fw_bk_idx'] == 0
                not_update_ma_teacher = False
        else:
            if not self.printed_loss_idx_info:
                logging.info('not selecting one task from {}'.format(self.current_tasks))
                self.printed_loss_idx_info = True
            not_update_ma_teacher = False

        if self.ma_teacher \
                and self.training \
                and batch.get('gradient_acc_start', True) \
                and not not_update_ma_teacher:
            self.update_ma_teacher()

        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        if 'itm_only' in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        if 'clip' in self.current_tasks:
            ret.update(objectives.compute_clip(self, batch))

        if 'fork_clip' in self.current_tasks:
            ret.update(objectives.compute_fork_clip(self, batch))

        if 'fork_ema_clip' in self.current_tasks:
            ret.update(objectives.compute_fork_ema_clip(self, batch))

        if 'seq2seq' in self.current_tasks:
            ret.update(objectives.compute_seq2seq(self, batch))

        if 'simclr' in self.current_tasks:
            ret.update(objectives.compute_simclr(self, batch))

        if 'refcont' in self.current_tasks:
            ret.update(objectives.compute_refcont(self, batch))

        if 'ema_clip' in self.current_tasks:
            ret.update(objectives.compute_ema_clip(self, batch))

        if 'decoder' in self.current_tasks:
            ret.update(objectives.compute_decoder(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)

