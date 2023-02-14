import torch
from captum.attr import IntegratedGradients, GradientShap, InputXGradient, Saliency, DeepLift

attr_algos = {
	'integrated-gradients' : IntegratedGradients,
	'gradient-shap' : GradientShap,
    'input-x-gradient': InputXGradient,
    'saliency': Saliency,
    'deep-lift': DeepLift,
}

baseline_required = {
	'integrated-gradients' : True,
	'gradient-shap': True,
    'input-x-gradient': False,
    'saliency': False,
    'deep-lift': True,
}


def calc_expl(attrs, k, attn_mask, min_val=-1e10):
    num_tokens = torch.sum(attn_mask, dim=1) - 1 # don't include CLS token when computing num_tokens
    num_highlight_tokens = torch.round(num_tokens * k / 100)
    ones = torch.ones_like(num_highlight_tokens)
    num_highlight_tokens = torch.maximum(num_highlight_tokens, ones).long()

    attrs = attrs + (1 - attn_mask) * min_val # ignore pad tokens when computing sorted_attrs_indices
    attrs[:, 0] = min_val # don't include CLS token when computing sorted_attrs_indices
    sorted_attrs_indices = torch.argsort(attrs, dim=1, descending=True)

    expl = torch.zeros_like(attn_mask).long()
    for i in range(len(attrs)):
        salient_indices = sorted_attrs_indices[i][:num_highlight_tokens[i]]
        expl[i, salient_indices] = 1
    expl[:, 0] = 1 # always treat CLS token as positive token

    return expl