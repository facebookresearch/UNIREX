import getpass, logging, socket
from typing import Any, List
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.loggers import NeptuneLogger
from src.utils.metrics import calc_preds, get_step_metrics, get_epoch_metrics

API_LIST = {
    "neptune": {
        'your-local-username': 'your-api-token',
    },
}


def get_username():
    return getpass.getuser()

def flatten_cfg(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        ret = {}
        for k, v in cfg.items():
            flatten: dict = flatten_cfg(v)
            ret.update({
                f"{k}/{f}" if f else k: fv
                for f, fv in flatten.items()
            })
        return ret
    return {"": cfg}

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def get_neptune_logger(
    cfg: DictConfig, project_name: str,
    name: str, tag_attrs: List[str], log_db: str,
    offline: bool, logger: str,
):
    neptune_api_key = API_LIST["neptune"][get_username()]

    # flatten cfg
    args_dict = {
        **flatten_cfg(OmegaConf.to_object(cfg)),
        "hostname": socket.gethostname()
    }
    tags = tag_attrs
    if cfg.model.expl_reg:
        tags.append('expl_reg')

    tags.append(log_db)

    neptune_logger = NeptuneLogger(
        api_key=neptune_api_key,
        project_name=project_name,
        experiment_name=name,
        params=args_dict,
        tags=tags,
        offline_mode=offline,
    )

    try:
        # for unknown reason, must access this field otherwise becomes None
        print(neptune_logger.experiment)
    except BaseException:
        pass

    return neptune_logger

def log_data_to_neptune(model_class, data, data_name, data_type, suffix, split, ret_dict=None, topk=None, detach_data=True):
    if topk:
        for i, k in enumerate(topk):
            model_class.log(f'{split}_{data_name}_{k}_{data_type}_{suffix}', data[i].detach(), prog_bar=True, sync_dist=(split != 'train'))
            if ret_dict is not None:
                ret_dict[f'{data_name}_{k}_{data_type}'] = data[i].detach() if detach_data else data[i]
    else:
        data_key = 'loss' if f'{data_name}_{data_type}' == 'total_loss' else f'{data_name}_{data_type}'
        model_class.log(f'{split}_{data_key}_{suffix}', data.detach(), prog_bar=True, sync_dist=(split != 'train'))
        if ret_dict is not None:
            ret_dict[data_key] = data.detach() if detach_data else data
    
    return ret_dict

def log_step_losses(model_class, loss_dict, ret_dict, do_expl_reg, split):
    ret_dict = log_data_to_neptune(model_class, loss_dict['loss'], 'total', 'loss', 'step', split, ret_dict, topk=None, detach_data=False)
    ret_dict = log_data_to_neptune(model_class, loss_dict['task_loss'], 'task', 'loss', 'step', split, ret_dict, topk=None)
    if do_expl_reg:
        ret_dict = log_data_to_neptune(model_class, loss_dict['expl_loss'], 'expl', 'loss', 'step', split, ret_dict, topk=None)
        if model_class.comp_wt > 0:
            ret_dict = log_data_to_neptune(model_class, loss_dict['comp_loss'], 'comp', 'loss', 'step', split, ret_dict, topk=None)
            ret_dict = log_data_to_neptune(model_class, loss_dict['comp_losses'], 'comp', 'loss', 'step', split, ret_dict, topk=model_class.topk[split])
        if model_class.suff_wt > 0:
            ret_dict = log_data_to_neptune(model_class, loss_dict['suff_loss'], 'suff', 'loss', 'step', split, ret_dict, topk=None)
            ret_dict = log_data_to_neptune(model_class, loss_dict['suff_losses'], 'suff', 'loss', 'step', split, ret_dict, topk=model_class.topk[split])
        if model_class.plaus_wt > 0 and loss_dict.get('plaus_loss'):
            ret_dict = log_data_to_neptune(model_class, loss_dict['plaus_loss'], 'plaus', 'loss', 'step', split, ret_dict, topk=None)
        if model_class.l2e and loss_dict.get('l2e_loss'):
            ret_dict = log_data_to_neptune(model_class, loss_dict['l2e_loss'], 'l2e', 'loss', 'step', split, ret_dict, topk=None)
        if model_class.a2r and loss_dict.get('a2r_loss'):
            ret_dict = log_data_to_neptune(model_class, loss_dict['a2r_loss'], 'a2r', 'loss', 'step', split, ret_dict, topk=None)
            
    return ret_dict

def log_step_metrics(model_class, metric_dict, ret_dict, split):
    ret_dict = log_data_to_neptune(model_class, metric_dict['comp_aopc'], 'comp_aopc', 'metric', 'step', split, ret_dict, topk=None)
    ret_dict = log_data_to_neptune(model_class, metric_dict['comps'], 'comp', 'metric', 'step', split, ret_dict, topk=model_class.topk[split])
    ret_dict = log_data_to_neptune(model_class, metric_dict['suff_aopc'], 'suff_aopc', 'metric', 'step', split, ret_dict, topk=None)
    ret_dict = log_data_to_neptune(model_class, metric_dict['suffs'], 'suff', 'metric', 'step', split, ret_dict, topk=model_class.topk[split])
    if model_class.log_odds:
        ret_dict = log_data_to_neptune(model_class, metric_dict['log_odds_aopc'], 'log_odds_aopc', 'metric', 'step', split, ret_dict, topk=None)
        ret_dict = log_data_to_neptune(model_class, metric_dict['log_odds'], 'log_odds', 'metric', 'step', split, ret_dict, topk=model_class.topk[split])
    ret_dict = log_data_to_neptune(model_class, metric_dict['comp_aopc']-metric_dict['suff_aopc'], 'csd_aopc', 'metric', 'step', split, ret_dict, topk=None)
    ret_dict = log_data_to_neptune(model_class, metric_dict['comps']-metric_dict['suffs'], 'csd', 'metric', 'step', split, ret_dict, topk=model_class.topk[split])
    if metric_dict.get('plaus_auprc'):
        ret_dict = log_data_to_neptune(model_class, metric_dict['plaus_auprc'], 'plaus_auprc', 'metric', 'step', split, ret_dict, topk=None)
        ret_dict = log_data_to_neptune(model_class, metric_dict['plaus_token_f1'], 'plaus_token_f1', 'metric', 'step', split, ret_dict, topk=None)
    return ret_dict

def log_epoch_losses(model_class, outputs, split):
    loss = torch.stack([x['loss'] for x in outputs]).mean()
    task_loss = torch.stack([x['task_loss'] for x in outputs]).mean()
    log_data_to_neptune(model_class, loss, 'total', 'loss', 'epoch', split, ret_dict=None, topk=None)
    log_data_to_neptune(model_class, task_loss, 'task', 'loss', 'epoch', split, ret_dict=None, topk=None)
    
    if model_class.expl_reg:
        assert len([x.get('expl_loss') for x in outputs if x is not None]) > 0
        
        expl_loss = torch.stack([x.get('expl_loss') for x in outputs if x is not None]).mean()
        log_data_to_neptune(model_class, expl_loss, 'expl', 'loss', 'epoch', split, ret_dict=None, topk=None)

        if model_class.comp_wt > 0:
            comp_loss = torch.stack([x.get('comp_loss') for x in outputs if x.get('comp_loss') is not None]).mean()
            log_data_to_neptune(model_class, comp_loss, 'comp', 'loss', 'epoch', split, ret_dict=None, topk=None)
            comp_losses = torch.stack([torch.stack([x.get(f'comp_{k}_loss') for x in outputs if x.get(f'comp_{k}_loss') is not None]).mean() for k in model_class.topk[split]])
            log_data_to_neptune(model_class, comp_losses, 'comp', 'loss', 'epoch', split, ret_dict=None, topk=model_class.topk[split])
        if model_class.suff_wt > 0:
            suff_loss = torch.stack([x.get('suff_loss') for x in outputs if x.get('suff_loss') is not None]).mean()
            log_data_to_neptune(model_class, suff_loss, 'suff', 'loss', 'epoch', split, ret_dict=None, topk=None)
            suff_losses = torch.stack([torch.stack([x.get(f'suff_{k}_loss') for x in outputs if x.get(f'suff_{k}_loss') is not None]).mean() for k in model_class.topk[split]])
            log_data_to_neptune(model_class, suff_losses, 'suff', 'loss', 'epoch', split, ret_dict=None, topk=model_class.topk[split])
        if model_class.plaus_wt > 0 and outputs[0].get('plaus_loss'):
            plaus_loss = torch.stack([x.get('plaus_loss') for x in outputs if x.get('plaus_loss') is not None]).mean()
            log_data_to_neptune(model_class, plaus_loss, 'plaus', 'loss', 'epoch', split, ret_dict=None, topk=None)
        if model_class.l2e and outputs[0].get('l2e_loss'):
            l2e_loss = torch.stack([x.get('l2e_loss') for x in outputs if x.get('l2e_loss') is not None]).mean()
            log_data_to_neptune(model_class, l2e_loss, 'l2e', 'loss', 'epoch', split, ret_dict=None, topk=None)
        if model_class.a2r and outputs[0].get('a2r_loss'):
            a2r_loss = torch.stack([x.get('a2r_loss') for x in outputs if x.get('a2r_loss') is not None]).mean()
            log_data_to_neptune(model_class, a2r_loss, 'a2r', 'loss', 'epoch', split, ret_dict=None, topk=None)

def log_epoch_metrics(model_class, outputs, split):
    logits = torch.cat([x['logits'] for x in outputs])
    targets = torch.cat([x['targets'] for x in outputs])
    preds = calc_preds(logits)

    perf_metrics = get_step_metrics(preds, targets, model_class.perf_metrics)
    perf_metrics = get_epoch_metrics(model_class.perf_metrics)

    log_data_to_neptune(model_class, perf_metrics['acc'], 'acc', 'metric', 'epoch', split, ret_dict=None, topk=None)
    log_data_to_neptune(model_class, perf_metrics['macro_f1'], 'macro_f1', 'metric', 'epoch', split, ret_dict=None, topk=None)
    log_data_to_neptune(model_class, perf_metrics['micro_f1'], 'micro_f1', 'metric', 'epoch', split, ret_dict=None, topk=None)
    if model_class.num_classes == 2:
        log_data_to_neptune(model_class, perf_metrics['binary_f1'], 'binary_f1', 'metric', 'epoch', split, ret_dict=None, topk=None)
    
    assert len([x.get('comp_aopc_metric') for x in outputs if x.get('comp_aopc_metric') is not None]) > 0
    comp_aopc = torch.stack([x.get('comp_aopc_metric') for x in outputs if x.get('comp_aopc_metric') is not None]).mean()
    comps = torch.stack([torch.stack([x.get(f'comp_{k}_metric') for x in outputs if x.get(f'comp_{k}_metric') is not None]).mean() for k in model_class.topk[split]])
    log_data_to_neptune(model_class, comp_aopc, 'comp_aopc', 'metric', 'epoch', split, ret_dict=None, topk=None)
    log_data_to_neptune(model_class, comps, 'comp', 'metric', 'epoch', split, ret_dict=None, topk=model_class.topk[split])

    assert len([x.get('suff_aopc_metric') for x in outputs if x.get('suff_aopc_metric') is not None]) > 0
    suff_aopc = torch.stack([x.get('suff_aopc_metric') for x in outputs if x.get('suff_aopc_metric') is not None]).mean()
    suffs = torch.stack([torch.stack([x.get(f'suff_{k}_metric') for x in outputs if x.get(f'suff_{k}_metric') is not None]).mean() for k in model_class.topk[split]])
    log_data_to_neptune(model_class, suff_aopc, 'suff_aopc', 'metric', 'epoch', split, ret_dict=None, topk=None)
    log_data_to_neptune(model_class, suffs, 'suff', 'metric', 'epoch', split, ret_dict=None, topk=model_class.topk[split])

    if model_class.log_odds:
        assert len([x.get('log_odds_aopc_metric') for x in outputs if x.get('log_odds_aopc_metric') is not None]) > 0
        log_odds_aopc = torch.stack([x.get('log_odds_aopc_metric') for x in outputs if x.get('log_odds_aopc_metric') is not None]).mean()
        log_odds = torch.stack([torch.stack([x.get(f'log_odds_{k}_metric') for x in outputs if x.get(f'log_odds_{k}_metric') is not None]).mean() for k in model_class.topk[split]])
        log_data_to_neptune(model_class, log_odds_aopc, 'log_odds_aopc', 'metric', 'epoch', split, ret_dict=None, topk=None)
        log_data_to_neptune(model_class, log_odds, 'log_odds', 'metric', 'epoch', split, ret_dict=None, topk=model_class.topk[split])

    csd_aopc = torch.stack([x.get('csd_aopc_metric') for x in outputs if x.get('csd_aopc_metric') is not None]).mean()
    csds = torch.stack([torch.stack([x.get(f'csd_{k}_metric') for x in outputs if x.get(f'csd_{k}_metric') is not None]).mean() for k in model_class.topk[split]])
    log_data_to_neptune(model_class, csd_aopc, 'csd_aopc', 'metric', 'epoch', split, ret_dict=None, topk=None)
    log_data_to_neptune(model_class, csds, 'csd', 'metric', 'epoch', split, ret_dict=None, topk=model_class.topk[split])
    
    if outputs[0].get('plaus_auprc_metric'):
        assert len([x.get('plaus_auprc_metric') for x in outputs if x.get('plaus_auprc_metric') is not None]) > 0
        plaus_auprc = torch.stack([x.get('plaus_auprc_metric') for x in outputs if x.get('plaus_auprc_metric') is not None]).mean()
        log_data_to_neptune(model_class, plaus_auprc, 'plaus_auprc', 'metric', 'epoch', split, ret_dict=None, topk=None)

        assert len([x.get('plaus_token_f1_metric') for x in outputs if x.get('plaus_token_f1_metric') is not None]) > 0
        plaus_token_f1 = torch.stack([x.get('plaus_token_f1_metric') for x in outputs if x.get('plaus_token_f1_metric') is not None]).mean()
        log_data_to_neptune(model_class, plaus_token_f1, 'plaus_token_f1', 'metric', 'epoch', split, ret_dict=None, topk=None)

    if 'delta' in outputs[0].keys():
        delta = torch.abs(torch.cat([x['delta'] for x in outputs])).mean()
        log_data_to_neptune(model_class, delta, 'convergence_delta', 'metric', 'epoch', split, ret_dict=None, topk=None)