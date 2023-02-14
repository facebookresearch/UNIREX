import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class BestPerformance(Callback):

    def __init__(self, monitor, mode):
        super().__init__()

        self.monitor = monitor
        assert monitor.split('_')[0] == 'dev'
        self.test_monitor = '_'.join(['test'] + monitor.split('_')[1:])

        self.mode = mode
        assert mode in ['max', 'min']

    def set_best_expl_metric(self, trainer, pl_module, metric):
        assert metric in ['comp', 'suff', 'log_odds', 'csd', 'plaus']
        for split in ['dev', 'test']:
            if metric == 'plaus':
                pl_module.best_metrics[f'{split}_best_{metric}_auprc'] = trainer.callback_metrics[f'{split}_{metric}_auprc_metric_epoch']
                pl_module.best_metrics[f'{split}_best_{metric}_token_f1'] = trainer.callback_metrics[f'{split}_{metric}_token_f1_metric_epoch']
            else:
                pl_module.best_metrics[f'{split}_best_{metric}_aopc'] = trainer.callback_metrics[f'{split}_{metric}_aopc_metric_epoch']
                for k in pl_module.topk[split]:
                    pl_module.best_metrics[f'{split}_best_{metric}_{k}'] = trainer.callback_metrics[f'{split}_{metric}_{k}_metric_epoch']
        return pl_module

    def log_best_expl_metric(self, pl_module, metric):
        assert metric in ['comp', 'suff', 'log_odds', 'csd', 'plaus']
        for split in ['dev', 'test']:
            if metric == 'plaus':
                pl_module.log(f'{split}_best_{metric}_auprc', pl_module.best_metrics[f'{split}_best_{metric}_auprc'], prog_bar=True, sync_dist=True)
                pl_module.log(f'{split}_best_{metric}_token_f1', pl_module.best_metrics[f'{split}_best_{metric}_token_f1'], prog_bar=True, sync_dist=True)
            else:
                pl_module.log(f'{split}_best_{metric}_aopc', pl_module.best_metrics[f'{split}_best_{metric}_aopc'], prog_bar=True, sync_dist=True)
                for k in pl_module.topk[split]:
                    pl_module.log(f'{split}_best_{metric}_{k}', pl_module.best_metrics[f'{split}_best_{metric}_{k}'], prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.mode == 'max':
            if pl_module.best_metrics['dev_best_perf'] == None:
                assert pl_module.best_metrics['test_best_perf'] == None
                pl_module.best_metrics['dev_best_perf'] = -float('inf')

            if trainer.callback_metrics[self.monitor] > pl_module.best_metrics['dev_best_perf']:
                pl_module.best_metrics['dev_best_perf'] = trainer.callback_metrics[self.monitor]
                pl_module.best_metrics['test_best_perf'] = trainer.callback_metrics[self.test_monitor]
                pl_module.best_metrics['best_epoch'] = trainer.current_epoch
                pl_module = self.set_best_expl_metric(trainer, pl_module, 'comp')
                pl_module = self.set_best_expl_metric(trainer, pl_module, 'suff')
                if pl_module.log_odds:
                    pl_module = self.set_best_expl_metric(trainer, pl_module, 'log_odds')
                pl_module = self.set_best_expl_metric(trainer, pl_module, 'csd')
                pl_module = self.set_best_expl_metric(trainer, pl_module, 'plaus')

        else:
            if pl_module.best_metrics['dev_best_perf'] == None:
                assert pl_module.best_metrics['test_best_perf'] == None
                pl_module.best_metrics['dev_best_perf'] = float('inf')

            if trainer.callback_metrics[self.monitor] < pl_module.best_metrics['dev_best_perf']:
                pl_module.best_metrics['dev_best_perf'] = trainer.callback_metrics[self.monitor]
                pl_module.best_metrics['test_best_perf'] = trainer.callback_metrics[self.test_monitor]
                pl_module.best_metrics['best_epoch'] = trainer.current_epoch
                pl_module = self.set_best_expl_metric(trainer, pl_module, 'comp')
                pl_module = self.set_best_expl_metric(trainer, pl_module, 'suff')
                if pl_module.log_odds:
                    pl_module = self.set_best_expl_metric(trainer, pl_module, 'log_odds')
                pl_module = self.set_best_expl_metric(trainer, pl_module, 'csd')
                pl_module = self.set_best_expl_metric(trainer, pl_module, 'plaus')

        pl_module.log('dev_best_perf', pl_module.best_metrics['dev_best_perf'], prog_bar=True, sync_dist=True)
        pl_module.log('test_best_perf', pl_module.best_metrics['test_best_perf'], prog_bar=True, sync_dist=True)
        pl_module.log('best_epoch', pl_module.best_metrics['best_epoch'], prog_bar=True, sync_dist=True)
        self.log_best_expl_metric(pl_module, 'comp')
        self.log_best_expl_metric(pl_module, 'suff')
        if pl_module.log_odds:
            self.log_best_expl_metric(pl_module, 'log_odds')
        self.log_best_expl_metric(pl_module, 'csd')
        self.log_best_expl_metric(pl_module, 'plaus')
