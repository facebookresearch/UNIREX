import os, pickle, warnings
from typing import Optional, List
from timeit import default_timer as timer

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from transformers import AutoModel, AutoTokenizer

from src.model.base_model import BaseModel
from src.model.mlp import MLP_factory
from src.utils.data import dataset_info
from src.utils.losses import calc_task_loss, calc_comp_loss, calc_suff_loss, calc_plaus_loss, calc_l2e_loss, calc_a2r_loss
from src.utils.metrics import init_best_metrics, init_perf_metrics, calc_preds, calc_comp, calc_suff, calc_log_odds, calc_aopc, calc_plaus
from src.utils.expl import attr_algos, baseline_required, calc_expl
from src.utils.optim import setup_optimizer_params, setup_scheduler, freeze_layers
from src.utils.logging import log_step_losses, log_step_metrics, log_epoch_losses, log_epoch_metrics


class LanguageModel(BaseModel):
    def __init__(self,
                 arch: str, dataset: str, optimizer: DictConfig, num_classes: int,
                 scheduler: DictConfig, num_freeze_layers: int = 0, freeze_epochs=-1, neg_weight=1,
                 expl_reg: bool = False, expl_reg_freq: int = 1, task_wt: float = None,
                 train_topk: List[int] = [1, 5, 10, 20, 50], eval_topk: List[int] = [1, 5, 10, 20, 50],
                 comp_wt: float = 0.0, comp_criterion: str = None, comp_margin: float = None, comp_target: bool = False,
                 suff_wt: float = 0.0, suff_criterion: str = None, suff_margin: float = None, suff_target: bool = False,
                 log_odds: bool = False, log_odds_target: bool = False,
                 plaus_wt: float = 0.0, plaus_criterion: str = None, plaus_margin: float = None,
                 explainer_type: str = None, expl_head_type: str = None, attr_algo: str = None, attr_pooling: str = None,
                 expl_head_mlp_hidden_dim: int = None, expl_head_mlp_hidden_layers: int = None, expl_head_mlp_dropout: float = 0.0, expl_head_mlp_layernorm: bool = False,
                 attr_mlp_hidden_dim: int = None, attr_mlp_hidden_layers: int = None, attr_mlp_dropout: float = 0.0, attr_mlp_layernorm: bool = False,
                 ig_steps: int = 3, internal_batch_size: int = None, return_convergence_delta: bool = False, gradshap_n_samples: int = 3, gradshap_stdevs: float = 0.0,
                 fresh: bool = False, fresh_extractor: str = None,
                 l2e: bool = False, l2e_wt: float = 0.0, l2e_criterion: str = None, l2e_classes: int = 5,
                 a2r: bool = False, a2r_wt: float = 0.0, a2r_criterion: str = None, a2r_task_out: str = None,
                 save_outputs: bool = False, exp_id: str = None,
                 measure_attrs_runtime: bool = False,
                 **kwargs):

        super().__init__()

        self.save_hyperparameters()

        self.arch = arch
        self.dataset = dataset
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.max_length = dataset_info[dataset]['max_length'][arch]

        self.scheduler = scheduler
        self.freeze_epochs = freeze_epochs
        self.neg_weight = neg_weight

        self.expl_reg = expl_reg
        self.expl_reg_freq = expl_reg_freq
        self.task_wt = task_wt
        
        assert len(train_topk) > 0 and all([0 < x <= 100 for x in train_topk])
        assert len(eval_topk) > 0 and all([0 < x <= 100 for x in eval_topk])
        self.topk = {'train': train_topk, 'dev': eval_topk, 'test': eval_topk}
        
        self.comp_wt = comp_wt
        self.comp_criterion = comp_criterion
        self.comp_margin = comp_margin
        self.comp_target = comp_target

        self.suff_wt = suff_wt
        self.suff_criterion = suff_criterion
        self.suff_margin = suff_margin
        self.suff_target = suff_target

        self.log_odds = log_odds
        self.log_odds_target = log_odds_target

        self.plaus_wt = plaus_wt
        self.plaus_criterion = plaus_criterion
        self.plaus_margin = plaus_margin

        self.best_metrics = init_best_metrics()
        self.perf_metrics = init_perf_metrics(num_classes)

        self.register_buffer('empty_tensor', torch.LongTensor([]))
        if num_classes == 2:
            self.register_buffer('class_weights', torch.FloatTensor([neg_weight, 1]))
        else:
            self.class_weights = None

        if fresh:
            assert explainer_type is None
        else:
            assert explainer_type in ['lm', 'self_lm', 'attr_algo']
        self.explainer_type = explainer_type

        assert attr_algo in list(attr_algos.keys()) + ['gold', 'inv', 'rand', None]
        self.attr_algo = attr_algo
        
        if expl_reg and not a2r:
            assert comp_criterion in ['diff', 'margin']
            assert comp_margin >= 0
            assert suff_criterion in ['diff', 'margin', 'mae', 'mse', 'kldiv']
            assert suff_margin >= 0
            assert plaus_criterion in ['margin', 'bce', 'kldiv', 'mse', 'mae']
            assert plaus_margin >= 0
        
        self.tokenizer = AutoTokenizer.from_pretrained(arch)

        self.attr_dict = {
            'explainer_type': explainer_type,
            'attr_algo': attr_algo,
            'attr_pooling': attr_pooling,
        }
        
        if explainer_type == 'attr_algo':
            assert attr_algo is not None
            self.task_encoder = AutoModel.from_pretrained(arch)
            self.task_head = nn.Linear(
                self.task_encoder.config.hidden_size,
                num_classes if self.dataset != 'cose' else 1
            )
            self.expl_encoder = None
            self.expl_head = None
            
            if attr_algo in attr_algos.keys():
                self.attr_dict['baseline_required'] = baseline_required[attr_algo]
                if attr_algo == 'integrated-gradients':
                    self.attr_dict['ig_steps'] = ig_steps
                    self.attr_dict['internal_batch_size'] = internal_batch_size
                    self.attr_dict['return_convergence_delta'] = return_convergence_delta
                elif attr_algo == 'gradient-shap':
                    self.attr_dict['gradshap_n_samples'] = gradshap_n_samples
                    self.attr_dict['gradshap_stdevs'] = gradshap_stdevs
                self.attr_dict['attr_func'] = attr_algos[attr_algo](self)
                self.attr_dict['tokenizer'] = self.tokenizer
                
                assert attr_pooling in ['sum', 'mlp']
                if attr_pooling == 'mlp':
                    assert expl_reg
                    self.attr_pooler = MLP_factory(
                        layer_sizes = [
                            [self.task_encoder.config.hidden_size, 1],
                            [attr_mlp_hidden_dim, attr_mlp_hidden_layers],
                            [1, 1],
                        ],
                        dropout=attr_mlp_dropout,
                        layernorm=attr_mlp_layernorm,
                    )

        elif self.explainer_type in ['lm', 'self_lm']:
            assert expl_reg
            if self.explainer_type == 'lm':
                self.task_encoder = AutoModel.from_pretrained(arch)
                self.expl_encoder = AutoModel.from_pretrained(arch)
            elif self.explainer_type == 'self_lm':
                self.encoder = AutoModel.from_pretrained(arch)
                self.task_encoder = self.encoder
                self.expl_encoder = self.encoder
            self.task_head = nn.Linear(
                self.task_encoder.config.hidden_size,
                num_classes if self.dataset != 'cose' else 1
            )

            assert expl_head_type in ['linear', 'mlp']
            expl_out_dim = l2e_classes if (l2e and l2e_criterion == 'ce') else 1
            if expl_head_type == 'linear':
                self.expl_head = nn.Linear(self.expl_encoder.config.hidden_size, expl_out_dim)
            elif expl_head_type == 'mlp':
                self.expl_head = MLP_factory(
                    layer_sizes = [
                        [self.expl_encoder.config.hidden_size, 1],
                        [expl_head_mlp_hidden_dim, expl_head_mlp_hidden_layers],
                        [expl_out_dim, 1],
                    ],
                    dropout=expl_head_mlp_dropout,
                    layernorm=expl_head_mlp_layernorm,
                )

            if a2r:
                assert a2r_task_out in ['sum', 'concat']
                a2r_task_factor = 2 if a2r_task_out == 'concat' else 1
                self.a2r_task_encoder = AutoModel.from_pretrained(arch)
                self.a2r_task_head = nn.Linear(
                    self.a2r_task_encoder.config.hidden_size * a2r_task_factor, 
                    num_classes if self.dataset != 'cose' else 1
                )

        elif fresh:
            assert attr_algo is None
            self.task_encoder = AutoModel.from_pretrained(arch)
            self.task_head = nn.Linear(
                self.task_encoder.config.hidden_size,
                num_classes if self.dataset != 'cose' else 1
            )
            self.expl_encoder = None
            self.expl_head = None

        else:
            raise NotImplementedError

        assert num_freeze_layers >= 0
        if num_freeze_layers > 0:
            freeze_layers(self, num_freeze_layers)

        self.model_dict = {
            'task_encoder': self.task_encoder,
            'task_head': self.task_head,
            'expl_encoder': self.expl_encoder,
            'expl_head': self.expl_head,
        }
        if explainer_type == 'attr_algo' and attr_algo in attr_algos.keys() and attr_pooling == 'mlp':
            self.model_dict['attr_pooler'] = self.attr_pooler
        elif a2r:
            self.model_dict['a2r_task_encoder'] = self.a2r_task_encoder
            self.model_dict['a2r_task_head'] = self.a2r_task_head

        self.fresh = fresh
        if fresh:
            assert not expl_reg
            assert not l2e
        self.fresh_extractor = fresh_extractor

        self.l2e = l2e
        if l2e:
            assert expl_reg
            assert not fresh
            assert l2e_wt > 0
            assert l2e_criterion is not None
            assert comp_wt == 0
            assert suff_wt == 0
            assert explainer_type in ['lm', 'self_lm']
        self.l2e_wt = l2e_wt
        self.l2e_criterion = l2e_criterion
        self.l2e_classes = l2e_classes

        self.a2r = a2r
        if a2r:
            assert expl_reg
            assert not fresh
            assert not l2e
            assert a2r_wt > 0
            assert a2r_criterion is not None
            assert comp_wt == 0
            assert suff_wt == 0
        self.a2r_wt = a2r_wt
        self.a2r_criterion = a2r_criterion
        self.a2r_task_out = a2r_task_out

        if save_outputs:
            assert exp_id is not None
        self.save_outputs = save_outputs
        self.exp_id = exp_id
        
        self.measure_attrs_runtime = measure_attrs_runtime


    def calc_attrs(self, input_ids, attn_mask, targets, rationale=None, noisy_rationale=None):
        # Compute attrs via grad-based attr algo
        if self.attr_dict['explainer_type'] == 'attr_algo' and self.attr_dict['attr_algo'] in attr_algos.keys():
            # If dataset is CoS-E, use zeros as targets
            if self.dataset == 'cose':
                targets = torch.zeros(len(input_ids)).long().to(targets.device)

            # Compute input embs and baseline embs
            input_embeds, baseline_embeds = self.get_attr_func_inputs(
                input_ids,
                self.attr_dict['baseline_required'],
            )

            # Compute dim-level attrs via attr algo
            if self.attr_dict['attr_algo'] == 'integrated-gradients':
                if self.attr_dict['return_convergence_delta']:
                    attrs, delta = self.attr_dict['attr_func'].attribute(
                        inputs=input_embeds.requires_grad_(), baselines=baseline_embeds,
                        target=targets, additional_forward_args=(attn_mask, 'captum'),
                        n_steps=self.attr_dict['ig_steps'], internal_batch_size=self.attr_dict['internal_batch_size'],
                        return_convergence_delta=self.attr_dict['return_convergence_delta'],
                    )
                    attrs, delta = attrs.float(), delta.float()
                else:
                    attrs = self.attr_dict['attr_func'].attribute(
                        inputs=input_embeds.requires_grad_(), baselines=baseline_embeds,
                        target=targets, additional_forward_args=(attn_mask, 'captum'),
                        n_steps=self.attr_dict['ig_steps'], internal_batch_size=self.attr_dict['internal_batch_size'],
                    ).float()
            elif self.attr_dict['attr_algo'] == 'gradient-shap':
                attrs = self.attr_dict['attr_func'].attribute(
                    inputs=input_embeds.requires_grad_(), baselines=baseline_embeds,
                    target=targets, additional_forward_args=(attn_mask, 'captum'),
                    n_samples=self.attr_dict['gradshap_n_samples'], stdevs=self.attr_dict['gradshap_stdevs'],
                ).float()
            elif self.attr_dict['attr_algo'] == 'deep-lift':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    attrs = self.attr_dict['attr_func'].attribute(
                        inputs=input_embeds.requires_grad_(), baselines=baseline_embeds,
                        target=targets, additional_forward_args=(attn_mask, 'captum'),
                    ).float()
            elif self.attr_dict['attr_algo'] in ['input-x-gradient', 'saliency']:
                attrs = self.attr_dict['attr_func'].attribute(
                    inputs=input_embeds.requires_grad_(),
                    target=targets, additional_forward_args=(attn_mask, 'captum'),
                ).float()

            # Pool dim-level attrs into token-level attrs
            if self.attr_dict['attr_pooling'] == 'sum':
                attrs = torch.sum(attrs, dim=-1)
            elif self.attr_dict['attr_pooling'] == 'mlp':
                attrs = self.attr_pooler(attrs).squeeze(-1)

        # Compute attrs via simple heuristic
        elif self.attr_dict['explainer_type'] == 'attr_algo':
            assert self.attr_dict['attr_algo'] in ['gold', 'inv', 'rand']
            if self.attr_dict['attr_algo'] == 'gold':
                attrs = rationale
            elif self.attr_dict['attr_algo'] in ['inv', 'rand']:
                assert noisy_rationale is not None
                attrs = noisy_rationale
            else:
                raise NotImplementedError

        # Compute attrs via LM
        elif self.attr_dict['explainer_type'] in ['lm', 'self_lm']:
            attrs = self.forward(input_ids, attn_mask, mode='expl')

        else:
            raise NotImplementedError

        # Mask out attrs for non-pad tokens
        if self.l2e:
            l2e_attrs = attrs * attn_mask.unsqueeze(2).expand(-1, -1, self.l2e_classes)
            attrs = torch.argmax(F.softmax(l2e_attrs, dim=2), dim=2) - int(self.l2e_classes / 2)
            attrs = attrs.float() * attn_mask
        else:
            l2e_attrs = None
            attrs = attrs * attn_mask

        # Make sure no attr scores are NaN
        assert not torch.any(torch.isnan(attrs))

        # Make sure no delta values are NaN
        if self.attr_dict['attr_algo'] == 'integrated-gradients':
            if self.attr_dict['return_convergence_delta']:
                assert self.attr_dict['attr_func'].has_convergence_delta()
                assert not torch.any(torch.isnan(delta))
        else:
            delta = None

        return attrs, l2e_attrs, delta

    def get_attr_func_inputs(self, input_ids, baseline_required):
        word_emb_layer = self.task_encoder.embeddings.word_embeddings
        tokenizer = self.attr_dict['tokenizer']
        input_embeds = word_emb_layer(input_ids)
        if baseline_required:
            baseline = torch.full(input_ids.shape, tokenizer.pad_token_id, device=input_ids.device).long()
            baseline[:, 0] = tokenizer.cls_token_id
            sep_token_locs = torch.nonzero(input_ids == tokenizer.sep_token_id)
            baseline[sep_token_locs[:, 0], sep_token_locs[:, 1]] = tokenizer.sep_token_id
            baseline_embeds = word_emb_layer(baseline)
        else:
            baseline_embeds = None
        return input_embeds, baseline_embeds

    def forward(self, inputs, attention_mask, mode='task', attrs=None):
        assert mode in ['task', 'expl', 'captum']

        if mode == 'task' and self.a2r and attrs is not None:
            a2r_attn_weights = F.softmax(attrs, dim=1).unsqueeze(2).expand(-1, -1, self.a2r_task_encoder.config.hidden_size)
            a2r_enc = self.a2r_task_encoder(input_ids=inputs, attention_mask=attention_mask).last_hidden_state
            if self.a2r_task_out == 'sum':
                a2r_enc = torch.sum(a2r_attn_weights * a2r_enc, dim=1)
            elif self.a2r_task_out == 'concat':
                a2r_enc = torch.cat(
                    (a2r_enc[:, 0, :], torch.sum(a2r_attn_weights * a2r_enc, dim=1)), dim=1
                )
            logits = self.a2r_task_head(a2r_enc)
        elif mode == 'task':
            enc = self.task_encoder(input_ids=inputs, attention_mask=attention_mask).pooler_output
            logits = self.task_head(enc)
            if self.dataset == 'cose':
                logits = logits.reshape(-1, self.num_classes)
        elif mode == 'expl':
            enc = self.expl_encoder(input_ids=inputs, attention_mask=attention_mask).last_hidden_state
            logits = self.expl_head(enc).squeeze(-1)
            if self.dataset == 'cose' and self.l2e:
                logits = logits.reshape(-1, self.max_length, self.num_classes)
            elif self.dataset == 'cose':
                logits = logits.reshape(-1, self.max_length)
        elif mode == 'captum':
            enc = self.task_encoder(inputs_embeds=inputs, attention_mask=attention_mask).pooler_output
            logits = self.task_head(enc)

        return logits

    def expl_forward(self, attrs, input_ids, attn_mask, targets, topk, expl_keys, mode, fresh=False, a2r=False):
        assert mode in ['loss', 'metric']
        batch_size, max_length = input_ids.shape
        if self.dataset == 'cose':
            batch_size = int(batch_size / self.num_classes)

        prev_end = 0
        expls = torch.stack([calc_expl(attrs, k, attn_mask) for k in topk]).reshape(-1, max_length)
        inv_expls = (1 - expls) * attn_mask.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
        inv_expls[:, 0] = 1 # always treat CLS token as positive token

        if 'task' in expl_keys:
            if fresh:
                fresh_input_ids, fresh_attn_mask = [], []
                for i, cur_attr in enumerate(attrs):
                    attr_nonzero = torch.nonzero(cur_attr).flatten()
                    num_pad_tokens = max_length-len(attr_nonzero)
                    fresh_input_ids.append(torch.cat((input_ids[i][attr_nonzero], self.tokenizer.pad_token_id*torch.ones(num_pad_tokens).long().to(input_ids.device))))
                    fresh_attn_mask.append(torch.cat((attn_mask[i][attr_nonzero], 0*torch.ones(num_pad_tokens).long().to(attn_mask.device))))

                fresh_input_ids = torch.stack(fresh_input_ids)
                fresh_attn_mask = torch.stack(fresh_attn_mask)
                input_ids_expand = fresh_input_ids
                attn_mask_expand = fresh_attn_mask
                task_start, task_end = prev_end, prev_end + batch_size
                prev_end = task_end

            elif a2r:
                if mode == 'loss':
                    a2r_input_ids = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                    a2r_attn_mask = expls
                elif mode == 'metric':
                    a2r_input_ids_ = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                    a2r_attn_mask_ = attn_mask.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                    a2r_input_ids, a2r_attn_mask = [], []
                    for i, cur_expl in enumerate(expls):
                        expls_nonzero = torch.nonzero(cur_expl).flatten()
                        num_pad_tokens = max_length-len(expls_nonzero)
                        a2r_input_ids.append(torch.cat((a2r_input_ids_[i][expls_nonzero], self.tokenizer.pad_token_id*torch.ones(num_pad_tokens).long().to(input_ids.device))))
                        a2r_attn_mask.append(torch.cat((a2r_attn_mask_[i][expls_nonzero], 0*torch.ones(num_pad_tokens).long().to(attn_mask.device))))
                    a2r_input_ids = torch.stack(a2r_input_ids)
                    a2r_attn_mask = torch.stack(a2r_attn_mask)
                
                input_ids_expand = a2r_input_ids
                attn_mask_expand = a2r_attn_mask
                task_start, task_end = prev_end, prev_end + batch_size*len(topk)
                prev_end = task_end
                
            else:
                input_ids_expand = input_ids
                attn_mask_expand = attn_mask
                task_start, task_end = prev_end, prev_end + batch_size
                prev_end = task_end

        else:
            attn_mask_expand = self.empty_tensor.clone()
            input_ids_expand = self.empty_tensor.clone()
            prev_end = 0
            
        if 'comp' in expl_keys:
            if mode == 'loss':
                comp_input_ids = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                input_ids_expand = torch.cat((input_ids_expand, comp_input_ids), dim=0)
                attn_mask_expand = torch.cat((attn_mask_expand, inv_expls), dim=0)
            elif mode == 'metric':
                if fresh:
                    comp_input_ids = torch.ones_like(fresh_input_ids) * self.tokenizer.pad_token_id
                    comp_attn_mask = torch.zeros_like(fresh_attn_mask)
                elif a2r:
                    comp_input_ids = torch.ones_like(a2r_input_ids) * self.tokenizer.pad_token_id
                    comp_attn_mask = torch.zeros_like(a2r_attn_mask)
                else:
                    comp_input_ids_ = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                    comp_attn_mask_ = attn_mask.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                    comp_input_ids, comp_attn_mask = [], []
                    for i, cur_inv_expl in enumerate(inv_expls):
                        inv_expls_nonzero = torch.nonzero(cur_inv_expl).flatten()
                        num_pad_tokens = max_length-len(inv_expls_nonzero)
                        comp_input_ids.append(torch.cat((comp_input_ids_[i][inv_expls_nonzero], self.tokenizer.pad_token_id*torch.ones(num_pad_tokens).long().to(input_ids.device))))
                        comp_attn_mask.append(torch.cat((comp_attn_mask_[i][inv_expls_nonzero], 0*torch.ones(num_pad_tokens).long().to(attn_mask.device))))
                    comp_input_ids = torch.stack(comp_input_ids)
                    comp_attn_mask = torch.stack(comp_attn_mask)
                    
                input_ids_expand = torch.cat((input_ids_expand, comp_input_ids), dim=0)
                attn_mask_expand = torch.cat((attn_mask_expand, comp_attn_mask), dim=0)

            comp_targets = None if targets is None else targets.unsqueeze(0).expand(len(topk), -1).reshape(-1)
            comp_start, comp_end = prev_end, prev_end + batch_size*len(topk)
            prev_end = comp_end

        else:
            comp_targets = None

        if 'suff' in expl_keys:
            if mode == 'loss':
                suff_input_ids = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                input_ids_expand = torch.cat((input_ids_expand, suff_input_ids), dim=0)
                attn_mask_expand = torch.cat((attn_mask_expand, expls), dim=0)
            elif mode == 'metric':
                if fresh:
                    suff_input_ids = fresh_input_ids
                    suff_attn_mask = fresh_attn_mask
                elif a2r:
                    suff_input_ids = a2r_input_ids
                    suff_attn_mask = a2r_attn_mask
                else:
                    suff_input_ids_ = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                    suff_attn_mask_ = attn_mask.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
                    suff_input_ids, suff_attn_mask = [], []
                    for i, cur_expl in enumerate(expls):
                        expls_nonzero = torch.nonzero(cur_expl).flatten()
                        num_pad_tokens = max_length-len(expls_nonzero)
                        suff_input_ids.append(torch.cat((suff_input_ids_[i][expls_nonzero], self.tokenizer.pad_token_id*torch.ones(num_pad_tokens).long().to(input_ids.device))))
                        suff_attn_mask.append(torch.cat((suff_attn_mask_[i][expls_nonzero], 0*torch.ones(num_pad_tokens).long().to(attn_mask.device))))
                    suff_input_ids = torch.stack(suff_input_ids)
                    suff_attn_mask = torch.stack(suff_attn_mask)
                    
                input_ids_expand = torch.cat((input_ids_expand, suff_input_ids), dim=0)
                attn_mask_expand = torch.cat((attn_mask_expand, suff_attn_mask), dim=0)

            suff_targets = None if targets is None else targets.unsqueeze(0).expand(len(topk), -1).reshape(-1)
            suff_start, suff_end = prev_end, prev_end + batch_size*len(topk)
            prev_end = suff_end

        else:
            suff_targets = None

        if 'log_odds' in expl_keys:
            assert mode == 'metric'
            log_odds_attn_mask = attn_mask.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
            log_odds_input_ids = input_ids.unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, max_length)
            log_odds_input_ids = log_odds_input_ids * inv_expls
            log_odds_input_ids = log_odds_input_ids + (self.tokenizer.pad_token_id * expls)
            if targets is None:
                log_odds_targets = None
            else:
                log_odds_targets = targets.unsqueeze(0).expand(len(topk), -1).reshape(-1)

            attn_mask_expand = torch.cat((attn_mask_expand, log_odds_attn_mask), dim=0)
            input_ids_expand = torch.cat((input_ids_expand, log_odds_input_ids), dim=0)

            log_odds_start, log_odds_end = prev_end, prev_end + batch_size*len(topk)
            prev_end = log_odds_end

        else:
            log_odds_targets = None

        logits_expand = self.forward(input_ids_expand.detach(), attn_mask_expand.detach())
        task_logits = logits_expand[task_start:task_end, :] if 'task' in expl_keys else None
        comp_logits = logits_expand[comp_start:comp_end, :] if 'comp' in expl_keys else None
        suff_logits = logits_expand[suff_start:suff_end, :] if 'suff' in expl_keys else None
        log_odds_logits = logits_expand[log_odds_start:log_odds_end, :] if 'log_odds' in expl_keys else None
        
        if a2r and mode == 'loss':
            assert targets is not None
            targets = targets.unsqueeze(0).expand(len(topk), -1).reshape(-1)
        elif a2r and mode == 'metric':
            task_logits = task_logits.reshape(len(topk), batch_size, self.num_classes)

        logits_dict = {
            'task': task_logits,
            'comp': comp_logits,
            'suff': suff_logits,
            'log_odds': log_odds_logits,
        }

        targets_dict = {
            'task': targets,
            'comp': comp_targets,
            'suff': suff_targets,
            'log_odds': log_odds_targets,
        }

        return logits_dict, targets_dict

    def run_step(self, batch, split, batch_idx):
        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        rationale = batch['rationale']
        has_rationale = batch['has_rationale']
        targets = batch['label']
        batch_size = len(input_ids)

        if self.dataset == 'cose':
            input_ids = input_ids.reshape(-1, self.max_length)
            attn_mask = attn_mask.reshape(-1, self.max_length)
            rationale = rationale.reshape(-1, self.max_length)
            has_rationale = has_rationale.reshape(-1)

        if self.attr_algo == 'inv':
            noisy_rationale = batch['inv_rationale']
        elif self.attr_algo == 'rand':
            noisy_rationale = batch['rand_rationale']
        else:
            noisy_rationale = None

        if self.dataset == 'cose' and noisy_rationale is not None:
            noisy_rationale = noisy_rationale.reshape(-1, self.max_length)

        if self.fresh:
            fresh_rationale = batch['fresh_rationale']
            if self.dataset == 'cose':
                fresh_rationale = fresh_rationale.reshape(-1, self.max_length)

        if self.l2e:
            assert batch['l2e_rationale'] is not None
            l2e_rationale = batch['l2e_rationale']

        eval_split: str = batch['split']
        if split == 'train':
            assert split == eval_split
        topk = self.topk[eval_split]
        ret_dict, loss_dict, metric_dict = {}, {}, {}

        do_expl_reg = self.expl_reg and (batch_idx % self.expl_reg_freq == 0)

        # Compute predictions and losses
        if do_expl_reg:
            # Compute attributions w.r.t. targets
            assert not self.fresh
            attrs, l2e_attrs, _ = self.calc_attrs(input_ids, attn_mask, targets, rationale, noisy_rationale)
            
            # Compute expl loss
            expl_loss = torch.tensor(0.0).to(self.device)
            if self.comp_wt > 0 or self.suff_wt > 0:
                logits_dict, targets_dict = self.expl_forward(attrs, input_ids, attn_mask, targets, topk, expl_keys=['task', 'comp', 'suff'], mode='loss')
                task_losses = calc_task_loss(logits_dict['task'], targets_dict['task'], reduction='none', class_weights=self.class_weights)
                logits = logits_dict['task']
                task_loss = self.task_wt * torch.mean(task_losses)
                task_losses = task_losses.unsqueeze(0).expand(len(topk), -1)
                
                if self.comp_wt > 0: # Compute comp loss
                    comp_losses = calc_comp_loss(
                        comp_logits=logits_dict['comp'],
                        comp_targets=targets_dict['comp'],
                        task_losses=task_losses,
                        comp_criterion=self.comp_criterion,
                        topk=topk,
                        comp_margin=self.comp_margin,
                    )
                    comp_loss = self.comp_wt * torch.mean(comp_losses)
                    expl_loss += comp_loss
                    loss_dict['comp_loss'] = comp_loss
                    loss_dict['comp_losses'] = comp_losses
                else:
                    loss_dict['comp_loss'] = torch.tensor(0.0).to(self.device)

                if self.suff_wt > 0: # Compute suff loss
                    suff_losses = calc_suff_loss(
                        suff_logits=logits_dict['suff'],
                        suff_targets=targets_dict['suff'],
                        task_losses=task_losses,
                        suff_criterion=self.suff_criterion,
                        topk=topk,
                        suff_margin=self.suff_margin,
                        task_logits = logits_dict['task'] if self.suff_criterion == 'kldiv' else None,
                    )
                    suff_loss = self.suff_wt * torch.mean(suff_losses)
                    expl_loss += suff_loss
                    loss_dict['suff_loss'] = suff_loss
                    loss_dict['suff_losses'] = suff_losses
                else:
                    loss_dict['suff_loss'] = torch.tensor(0.0).to(self.device)

            elif self.a2r:
                logits_dict, targets_dict = self.expl_forward(attrs, input_ids, attn_mask, targets, topk, expl_keys=['task'], mode='loss', a2r=self.a2r)
                logits, targets = logits_dict['task'], targets_dict['task']
                task_loss = self.task_wt * calc_task_loss(logits, targets)
                a2r_logits = self.forward(input_ids, attn_mask, attrs=attrs).unsqueeze(0).expand(len(topk), -1, -1).reshape(-1, self.num_classes)
            else:
                logits = self.forward(input_ids, attn_mask)
                task_loss = self.task_wt * calc_task_loss(logits, targets)

            if self.plaus_wt > 0 and rationale is not None: # Compute plaus loss
                plaus_loss = self.plaus_wt * calc_plaus_loss(
                    attrs=attrs,
                    rationale=rationale,
                    attn_mask=attn_mask,
                    plaus_criterion=self.plaus_criterion,
                    plaus_margin=self.plaus_margin,
                    has_rationale=has_rationale,
                )
                expl_loss += plaus_loss
                loss_dict['plaus_loss'] = plaus_loss

            if self.l2e_wt > 0 and l2e_rationale is not None: # Compute l2e loss
                l2e_loss = self.l2e_wt * calc_l2e_loss(
                    l2e_attrs=l2e_attrs,
                    l2e_rationale=l2e_rationale,
                    attn_mask=attn_mask,
                    l2e_criterion=self.l2e_criterion,
                )
                expl_loss += l2e_loss
                loss_dict['l2e_loss'] = l2e_loss

            if self.a2r_wt > 0 and a2r_logits is not None: # Compute a2r loss
                a2r_loss = self.a2r_wt * calc_a2r_loss(
                    logits=logits_dict['task'],
                    a2r_logits=a2r_logits,
                    a2r_criterion=self.a2r_criterion,
                )
                expl_loss += a2r_loss
                loss_dict['a2r_loss'] = a2r_loss

            loss = task_loss + expl_loss # Compute total loss
            loss_dict['expl_loss'] = expl_loss

        else:
            logits = self.forward(input_ids, fresh_rationale) if self.fresh else self.forward(input_ids, attn_mask)
            task_loss = calc_task_loss(logits, targets)
            loss = task_loss

        loss_dict['task_loss'] = task_loss
        loss_dict['loss'] = loss

        # Compute expl metrics
        with torch.no_grad():
            # Compute preds
            preds = calc_preds(logits)

            # Set models to eval mode
            for model in self.model_dict.values():
                if model is not None:
                    model.eval()

            # Compute attributions w.r.t. preds
            if self.measure_attrs_runtime and eval_split == 'test':
                start = timer()

            if self.fresh:
                attrs = fresh_rationale
            else:
                attrs, _, delta = self.calc_attrs(input_ids, attn_mask, preds, rationale, noisy_rationale)
                if self.attr_dict['attr_algo'] == 'integrated-gradients':
                    if self.attr_dict['return_convergence_delta']:
                        ret_dict['delta'] = delta.detach()
            
            if self.measure_attrs_runtime and eval_split == 'test':
                end = timer()
                batch_attrs_runtime = end - start
                attrs_runtime = batch_attrs_runtime / batch_size
                self.log(f'{eval_split}_attrs_runtime', attrs_runtime, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            
            # Perform comp/suff forward pass
            expl_keys = ['task', 'comp', 'suff']
            if self.log_odds:
                expl_keys.append('log_odds')
            logits_dict, _ = self.expl_forward(attrs, input_ids, attn_mask, None, topk, expl_keys=expl_keys, mode='metric', fresh=self.fresh, a2r=self.a2r)

        comp_logits = logits_dict['comp'].reshape(len(topk), batch_size, self.num_classes)
        if self.a2r:
            comp_targets = targets.reshape(len(topk), batch_size)
            metric_dict['comps'] = torch.stack([calc_comp(logits_dict['task'][i], comp_logits[i], comp_targets[i], self.comp_target) for i, k in enumerate(topk)])
        else:
            metric_dict['comps'] = torch.stack([calc_comp(logits_dict['task'], comp_logits[i], targets, self.comp_target) for i, k in enumerate(topk)])
        metric_dict['comp_aopc'] = calc_aopc(metric_dict['comps'])

        suff_logits = logits_dict['suff'].reshape(len(topk), batch_size, self.num_classes)
        if self.a2r:
            suff_targets = targets.reshape(len(topk), batch_size)
            metric_dict['suffs'] = torch.stack([calc_suff(logits_dict['task'][i], suff_logits[i], suff_targets[i], self.suff_target) for i, k in enumerate(topk)])
        else:
            metric_dict['suffs'] = torch.stack([calc_suff(logits_dict['task'], suff_logits[i], targets, self.suff_target) for i, k in enumerate(topk)])
        metric_dict['suff_aopc'] = calc_aopc(metric_dict['suffs'])

        if self.log_odds:
            log_odds_logits = logits_dict['log_odds'].reshape(len(topk), batch_size, self.num_classes)
            if self.a2r:
                log_odds_targets = targets.reshape(len(topk), batch_size)
                metric_dict['log_odds'] = torch.stack([calc_log_odds(logits_dict['task'][i], log_odds_logits[i], log_odds_targets[i], self.log_odds_target) for i, k in enumerate(topk)])
            else:
                metric_dict['log_odds'] = torch.stack([calc_log_odds(logits_dict['task'], log_odds_logits[i], targets, self.log_odds_target) for i, k in enumerate(topk)])
            metric_dict['log_odds_aopc'] = calc_aopc(metric_dict['log_odds'])

        if rationale is not None:
            metric_dict['plaus_auprc'], metric_dict['plaus_token_f1'] = calc_plaus(rationale, attrs, attn_mask, has_rationale)

        # Log step losses
        ret_dict = log_step_losses(self, loss_dict, ret_dict, do_expl_reg, eval_split)
        ret_dict['logits'] = logits.detach()
        ret_dict['targets'] = targets.detach()
        ret_dict['eval_split'] = eval_split

        # Log step metrics
        ret_dict = log_step_metrics(self, metric_dict, ret_dict, eval_split)

        # Save attrs
        if self.save_outputs:
            ret_dict['attrs'] = attrs.detach()

        return ret_dict

    def aggregate_epoch(self, outputs, split):
        if split == 'train':
            splits = ['train']
        elif split == 'dev':
            splits = ['dev', 'test']
        elif split == 'test':
            splits = [outputs[0]['eval_split']]
        outputs_list = outputs if split == 'dev' else [outputs]
        
        for dataset_idx, eval_split in enumerate(splits):
            outputs = outputs_list[dataset_idx]
            log_epoch_losses(self, outputs, eval_split) # Log epoch losses
            log_epoch_metrics(self, outputs, eval_split) # Log epoch metrics

        # Save outputs to file            
        if self.save_outputs:
            out_dir = f'{get_original_cwd()}/../save/{self.exp_id}/model_outputs/{self.dataset}'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            keys = ['preds', 'attrs']
            for key in keys:
                if key == 'preds':
                    logits = torch.cat([x['logits'] for x in outputs])
                    out_data = calc_preds(logits)
                else:
                    out_data = torch.cat([x[key] for x in outputs])
                out_data = out_data.cpu().detach()
                out_file = os.path.join(out_dir, f'{eval_split}_{key}.pkl')
                pickle.dump(out_data.squeeze(), open(out_file, 'wb'))

    def configure_optimizers(self):
        optimizer_params = setup_optimizer_params(self.model_dict, self.optimizer, self.explainer_type, self.attr_dict['attr_pooling'], self.a2r)
        self.optimizer['lr'] = self.optimizer['lr'] * self.trainer.world_size
        optimizer = instantiate(
            self.optimizer, params=optimizer_params,
            _convert_="partial"
        )
        if self.scheduler.lr_scheduler == 'linear_with_warmup':
            scheduler = setup_scheduler(self.scheduler, self.total_steps, optimizer)
            return [optimizer], [scheduler]
        elif self.lr_scheduler == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError