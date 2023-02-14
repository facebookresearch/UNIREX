import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import precision_recall_curve, auc, f1_score, average_precision_score


def init_best_metrics():
    return {
        'best_epoch': 0,
        'dev_best_perf': None,
        'test_best_perf': None,
    }

def init_perf_metrics(num_classes):
    perf_metrics = torch.nn.ModuleDict({
        'acc': torchmetrics.Accuracy(),
        'macro_f1': torchmetrics.F1(num_classes=num_classes, average='macro'),
        'micro_f1': torchmetrics.F1(num_classes=num_classes, average='micro'),
    })

    assert num_classes >= 2
    if num_classes == 2:
        perf_metrics['binary_f1'] = torchmetrics.F1(num_classes=num_classes, average='micro', ignore_index=0)
    
    return perf_metrics

def calc_preds(logits):
    return torch.argmax(logits, dim=1)

def calc_comp(logits, inv_expl_logits, targets=None, comp_target=False):
    assert not (comp_target and targets is None)
    preds = targets if comp_target else calc_preds(logits)

    probs = F.softmax(logits, dim=1)
    pred_probs = torch.gather(probs, dim=1, index=preds.unsqueeze(1)).flatten()

    inv_expl_probs = F.softmax(inv_expl_logits, dim=1)
    pred_inv_expl_probs = torch.gather(inv_expl_probs, dim=1, index=preds.unsqueeze(1)).flatten()
    
    return torch.mean(pred_probs - pred_inv_expl_probs)

def calc_suff(logits, expl_logits, targets=None, suff_target=False):
    assert not (suff_target and targets is None)
    preds = targets if suff_target else calc_preds(logits)

    probs = F.softmax(logits, dim=1)
    pred_probs = torch.gather(probs, dim=1, index=preds.unsqueeze(1)).flatten()

    expl_probs = F.softmax(expl_logits, dim=1)
    pred_expl_probs = torch.gather(expl_probs, dim=1, index=preds.unsqueeze(1)).flatten()

    return torch.mean(pred_probs - pred_expl_probs)

def calc_log_odds(logits, log_odds_logits, targets=None, log_odds_target=False):
    assert not (log_odds_target and targets is None)
    preds = targets if log_odds_target else calc_preds(logits)

    probs = -F.log_softmax(logits, dim=1)
    pred_probs = torch.gather(probs, dim=1, index=preds.unsqueeze(1)).flatten()

    log_odds_probs = -F.log_softmax(log_odds_logits, dim=1)
    pred_log_odds_probs = torch.gather(log_odds_probs, dim=1, index=preds.unsqueeze(1)).flatten()
    
    return torch.mean(pred_probs - pred_log_odds_probs)

def calc_aopc(values):
    return torch.sum(values) / (len(values)+1)

def calc_plaus(rationale, attrs, attn_mask, has_rationale, bin_thresh=0.0):
    batch_size = len(rationale)
    auprc_list, ap_list, token_f1_list = [], [], []
    for i in range(batch_size):
        if has_rationale[i] == 1:
            num_tokens = attn_mask[i].sum()
            assert torch.sum(rationale[i][:num_tokens]) > 0

            rationale_ = rationale[i][:num_tokens].detach().cpu().numpy()
            attrs_ = attrs[i][:num_tokens].detach().cpu().numpy()
            bin_attrs_ = (attrs_ > bin_thresh).astype('float32')

            precision, recall, _ = precision_recall_curve(
                y_true=rationale_,
                probas_pred=attrs_,
            )
            auprc_list.append(auc(recall, precision))

            token_f1 = f1_score(
                y_true=rationale_,
                y_pred=bin_attrs_,
                average='macro',
            )
            token_f1_list.append(token_f1)

        else:
            auprc_list.append(0.0)
            token_f1_list.append(0.0)
            ap_list.append(0.0)

    plaus_auprc = torch.tensor(np.mean(auprc_list))
    plaus_token_f1 = torch.tensor(np.mean(token_f1_list))

    return plaus_auprc, plaus_token_f1

def get_step_metrics(preds, targets, metrics):
    res = {}
    for key, metric_fn in metrics.items():
        res.update({key: metric_fn(preds, targets) * 100})
    return res

def get_epoch_metrics(metrics):
    res = {}
    for key, metric_fn in metrics.items():
        res.update({key: metric_fn.compute() * 100})
        metric_fn.reset()
    return res