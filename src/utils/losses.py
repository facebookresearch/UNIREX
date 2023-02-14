import torch
import torch.nn.functional as F


def calc_task_loss(logits, targets, reduction='mean', class_weights=None):
    assert len(logits) == len(targets)
    return F.cross_entropy(logits, targets, weight=class_weights, reduction=reduction)

def calc_comp_loss(comp_logits, comp_targets, task_losses, comp_criterion, topk, comp_margin=None):
    inv_expl_losses = calc_task_loss(comp_logits, comp_targets, reduction='none').reshape(len(topk), -1)
    if comp_criterion == 'diff':
        comp_losses = task_losses - inv_expl_losses
    elif comp_criterion == 'margin':
        assert comp_margin is not None
        comp_margins = comp_margin * torch.ones_like(inv_expl_losses)
        comp_losses = torch.maximum(-comp_margins, task_losses - inv_expl_losses) + comp_margins
    else:
        raise NotImplementedError

    assert not torch.any(torch.isnan(comp_losses))
    return torch.mean(comp_losses, dim=1)

def calc_suff_loss(suff_logits, suff_targets, task_losses, suff_criterion, topk, suff_margin=None, task_logits=None):
    if suff_criterion == 'kldiv':
        assert task_logits is not None
        batch_size = len(task_logits)
        task_distr = F.log_softmax(task_logits, dim=1).unsqueeze(0).expand(len(topk), -1, -1).reshape(len(topk) * batch_size, -1)
        suff_distr = F.softmax(suff_logits, dim=1)
        suff_losses = F.kl_div(task_distr, suff_distr, reduction='none').reshape(len(topk), -1)
    else:
        expl_losses = calc_task_loss(suff_logits, suff_targets, reduction='none').reshape(len(topk), -1)
        if suff_criterion == 'diff':
            suff_losses = expl_losses - task_losses
        elif suff_criterion == 'margin':
            suff_margins = suff_margin * torch.ones_like(expl_losses)
            suff_losses = torch.maximum(-suff_margins, expl_losses - task_losses) + suff_margins
        elif suff_criterion == 'mae':
            suff_losses = F.l1_loss(expl_losses, task_losses, reduction='none')
        elif suff_criterion == 'mse':
            suff_losses = F.mse_loss(expl_losses, task_losses, reduction='none')
        else:
            raise NotImplementedError

    assert not torch.any(torch.isnan(suff_losses))
    return torch.mean(suff_losses, dim=1)

def calc_plaus_loss(attrs, rationale, attn_mask, plaus_criterion, plaus_margin=None, has_rationale=None):
    if plaus_criterion == 'margin':
        raise NotImplementedError
        plaus_margins = attn_mask * plaus_margin
        inv_rationale = (1 - rationale) * attn_mask
        plaus_loss = (-rationale + inv_rationale) * attrs
        assert not torch.any(torch.isnan(plaus_loss))
        plaus_loss = torch.maximum(-plaus_margins, plaus_loss) + plaus_margins
        plaus_loss = torch.sum(plaus_loss) / torch.sum(attn_mask)
    elif plaus_criterion == 'bce':
        assert has_rationale is not None
        max_length = attn_mask.shape[1]
        has_rationale_ = has_rationale.unsqueeze(1).repeat(1, max_length) * attn_mask
        rationale = rationale * has_rationale_
        num_tokens = has_rationale_.sum()
        plaus_pos_wt = (num_tokens - rationale.sum()) / rationale.sum()
        plaus_loss = (F.binary_cross_entropy_with_logits(attrs, rationale, pos_weight=plaus_pos_wt, reduction='none') * has_rationale_).sum()
        if num_tokens > 0:
            plaus_loss /= num_tokens
        else:
            assert plaus_loss == 0
        assert not torch.any(torch.isnan(plaus_loss))
    else:
        raise NotImplementedError

    assert not torch.isnan(plaus_loss)
    return plaus_loss

def calc_l2e_loss(l2e_attrs, l2e_rationale, attn_mask, l2e_criterion):
    if l2e_criterion == 'ce':
        num_tokens = attn_mask.sum()
        num_classes = l2e_attrs.shape[2]
        l2e_attrs = l2e_attrs.reshape(-1, num_classes)
        l2e_loss = (F.cross_entropy(l2e_attrs, l2e_rationale.flatten(), reduction='none') * attn_mask.flatten()).sum()
        if num_tokens > 0:
            l2e_loss /= num_tokens
        else:
            assert l2e_loss == 0
        assert not torch.any(torch.isnan(l2e_loss))
    else:
        raise NotImplementedError

    assert not torch.isnan(l2e_loss)
    return l2e_loss

def calc_a2r_loss(logits, a2r_logits, a2r_criterion):
    if a2r_criterion == 'jsd':
        a2r_loss = js_div(logits, a2r_logits)
    else:
        raise NotImplementedError
    assert not torch.isnan(a2r_loss)
    return a2r_loss

def js_div(logits_1, logits_2, reduction='batchmean'):
    probs_m = (F.softmax(logits_1, dim=1) + F.softmax(logits_2, dim=1)) / 2.0
    
    loss_1 = F.kl_div(
        F.log_softmax(logits_1, dim=1),
        probs_m,
        reduction=reduction
    )
    
    loss_2 = F.kl_div(
        F.log_softmax(logits_2, dim=1),
        probs_m,
        reduction=reduction
    )

    loss = 0.5 * (loss_1 + loss_2)

    return loss