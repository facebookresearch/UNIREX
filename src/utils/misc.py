import torch


def clear_cache():
    torch.cuda.empty_cache()

def is_subsequence(a, b):
    # Check if list a is a subsequence if list b
    return any(a == b[i:i + len(a)] for i in range(len(b) - len(a) + 1))