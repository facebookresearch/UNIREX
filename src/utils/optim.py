from transformers import get_scheduler

no_decay = ['bias', 'LayerNorm.weight']


def setup_optimizer_params(model_dict, optimizer, explainer_type, attr_pooling=None, a2r=False):
    optimizer_parameters = [
        {
            'params': [p for n, p in model_dict['task_encoder'].named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': optimizer.weight_decay,
        },
        {
            'params': [p for n, p in model_dict['task_encoder'].named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
        {
            'params': [p for n, p in model_dict['task_head'].named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': optimizer.weight_decay,
        },
        {
            'params': [p for n, p in model_dict['task_head'].named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    if explainer_type == 'lm':
        optimizer_parameters += [
            {
                'params': [p for n, p in model_dict['expl_encoder'].named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': optimizer.weight_decay,
            },
            {
                'params': [p for n, p in model_dict['expl_encoder'].named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
            {
                'params': [p for n, p in model_dict['expl_head'].named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': optimizer.weight_decay,
            },
            {
                'params': [p for n, p in model_dict['expl_head'].named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
    elif explainer_type == 'self_lm':
        optimizer_parameters += [
            {
                'params': [p for n, p in model_dict['expl_head'].named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': optimizer.weight_decay,
            },
            {
                'params': [p for n, p in model_dict['expl_head'].named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
    elif explainer_type == 'attr_algo' and attr_pooling == 'mlp':
        optimizer_parameters += [
            {
                'params': [p for n, p in model_dict['attr_pooler'].named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': optimizer.weight_decay,
            },
            {
                'params': [p for n, p in model_dict['attr_pooler'].named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

    if a2r:
        optimizer_parameters += [
            {
                'params': [p for n, p in model_dict['a2r_task_encoder'].named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': optimizer.weight_decay,
            },
            {
                'params': [p for n, p in model_dict['a2r_task_encoder'].named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
            {
                'params': [p for n, p in model_dict['a2r_task_head'].named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': optimizer.weight_decay,
            },
            {
                'params': [p for n, p in model_dict['a2r_task_head'].named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

    return optimizer_parameters

def setup_scheduler(scheduler, total_steps, optimizer):
    if scheduler.warmup_updates > 1.0:
        warmup_steps = int(scheduler.warmup_updates)
    else:
        warmup_steps = int(total_steps *
                            scheduler.warmup_updates)
    print(
        f'\nTotal steps: {total_steps} with warmup steps: {warmup_steps}\n')

    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scheduler = {
        'scheduler': scheduler,
        'interval': 'step',
        'frequency': 1
    }
    return scheduler

def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze_layers(model, num_freeze_layers):
    if model.arch == 'google/bigbird-roberta-base':
        assert model.task_encoder is not None

        # Freeze task encoder's embedding layer
        for p in model.task_encoder.embeddings.parameters():
            p.requires_grad = False
        
        # Freeze task encoder's encoder layers
        for i in range(num_freeze_layers):
            for p in model.task_encoder.encoder.layer[i].parameters():
                p.requires_grad = False

        if model.expl_encoder is not None:
            # Freeze expl encoder's embedding layer
            for p in model.expl_encoder.embeddings.parameters():
                p.requires_grad = False
            
            # Freeze expl encoder's encoder layers
            for i in range(num_freeze_layers):
                for p in model.expl_encoder.encoder.layer[i].parameters():
                    p.requires_grad = False

    else:
        raise NotImplementedError