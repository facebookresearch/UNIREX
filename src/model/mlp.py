import torch.nn as nn

def MLP_factory(layer_sizes, dropout=False, layernorm=False):
    modules = nn.ModuleList()
    unpacked_sizes = []
    for block in layer_sizes:
        unpacked_sizes.extend([block[0]] * block[1])

    for k in range(len(unpacked_sizes)-1):
        if layernorm:
            modules.append(nn.LayerNorm(unpacked_sizes[k]))
        modules.append(nn.Linear(unpacked_sizes[k], unpacked_sizes[k+1]))
        if k < len(unpacked_sizes)-2:
            modules.append(nn.ReLU())
            if dropout is not False:
                modules.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*modules)
    return mlp