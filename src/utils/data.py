dataset_info = {
    'amazon': {
        'train': ['train', 10000],
        'dev': ['dev', 2000],
        'test': ['test', 2000],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 256,
            'google/bigbird-roberta-base': 256,
        },
        'num_special_tokens': 2,
    },
    'cose': {
        'train': ['train', 8752],
        'dev': ['val', 1086],
        'test': ['test', 1079],
        'num_classes': 5,
        'classes': ['A', 'B', 'C', 'D', 'E'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 77,
        },
        'num_special_tokens': None,
    },
    'esnli': {
        'train': ['train', 549309],
        'dev': ['val', 9823],
        'test': ['test', 9807],
        'num_classes': 3,
        'classes': ['entailment', 'neutral', 'contradiction'],
        'max_length': {
            'bert-base-uncased': 125,
            'google/bigbird-roberta-base': 125,
        },
        'num_special_tokens': 3,
    },
    'movies': {
        'train': ['train', 1599],
        'dev': ['val', 200],
        'test': ['test', 200],
        'num_classes': 2,
        'classes': ['NEG', 'POS'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 1024,
        },
        'num_special_tokens': 2,
    },
    'multirc': {
        'train': ['train', 24029],
        'dev': ['val', 3214],
        'test': ['test', 4848],
        'num_classes': 2,
        'classes': ['False', 'True'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 748,
        },
        'num_special_tokens': 3,
    },
    'sst': {
        'train': ['train', 6920],
        'dev': ['dev', 872],
        'test': ['test', 1821],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 58,
            'google/bigbird-roberta-base': 67,
        },
        'num_special_tokens': 2,
    },
    'stf': {
        'train': ['train', 7896],
        'dev': ['dev', 978],
        'test': ['test', 1998],
        'num_classes': 2,
        'classes': ['not_hate', 'hate'],
        'max_length': {
            'bert-base-uncased': 128,
            'google/bigbird-roberta-base': 128,
        },
        'num_special_tokens': 2,
    },
    'yelp': {
        'train': ['train', 10000],
        'dev': ['dev', 2000],
        'test': ['test', 2000],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 512,
        },
        'num_special_tokens': 2,
    },
    'olid': {
        'train': ['train', 11916],
        'dev': ['validation', 1324],
        'test': ['test', 860],
        'num_classes': 2,
        'classes': ['not_offensive', 'offensive'],
        'max_length': {
            'bert-base-uncased': 128,
            'google/bigbird-roberta-base': 128,
        },
        'num_special_tokens': 2,
    },
    'irony': {
        'train': ['train', 2862],
        'dev': ['validation', 955],
        'test': ['test', 784],
        'num_classes': 2,
        'classes': ['not_irony', 'irony'],
        'max_length': {
            'bert-base-uncased': 128,
            'google/bigbird-roberta-base': 128,
        },
        'num_special_tokens': 2,
    },
}

eraser_datasets = ['cose', 'esnli', 'movies', 'multirc']

monitor_dict = {
    'cose': 'dev_acc_metric_epoch',
    'esnli': 'dev_macro_f1_metric_epoch',
    'movies': 'dev_macro_f1_metric_epoch',
    'multirc': 'dev_macro_f1_metric_epoch',
    'sst': 'dev_acc_metric_epoch',
    'amazon': 'dev_acc_metric_epoch',
    'yelp': 'dev_acc_metric_epoch',
    'stf': 'dev_binary_f1_metric_epoch',
    'olid': 'dev_macro_f1_metric_epoch',
    'irony': 'dev_binary_f1_metric_epoch',
}

data_keys = ['item_idx', 'input_ids', 'attention_mask', 'rationale', 'inv_rationale', 'rand_rationale', 'has_rationale', 'label', 'rationale_indices']