"""
handle changes to the hydra config
"""
import time
import uuid
from pathlib import Path

import rich.syntax
import rich.tree
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def fail_on_missing(cfg: DictConfig) -> None:
    if isinstance(cfg, ListConfig):
        for x in cfg:
            fail_on_missing(x)
    elif isinstance(cfg, DictConfig):
        for _, v in cfg.items():
            fail_on_missing(v)


def pretty_print(
        cfg: DictConfig,
        fields=(
            "dataset",
            "model",
            "logger",
            "trainer",
            "setup",
            "training",
        )
):
    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = cfg.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=True)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # others defined in root
    others = tree.add("others", style=style, guide_style=style)
    for var, val in OmegaConf.to_container(cfg, resolve=True).items():
        if not var.startswith("_") and var not in fields:
            others.add(f"{var}: {val}")

    rich.print(tree)


@rank_zero_only
def touch(cfg: DictConfig) -> None:
    # sanity check
    assert Path(cfg.data.data_path).exists(), f"datapath {cfg.data.data_path} not exist"

    if cfg.training.finetune_ckpt:
        assert cfg.training.ckpt_path
    if cfg.training.evaluate_ckpt:
        assert cfg.training.ckpt_path
        assert cfg.training.eval_splits != "all"

    cfg.logger.name = f'{cfg.model.model}_{cfg.data.dataset}_{cfg.model.arch}_{time.strftime("%d_%m_%Y")}_{str(uuid.uuid4())[: 8]}'

    if cfg.debug:
        # for DEBUG purposes only
        cfg.trainer.limit_train_batches = 1
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.limit_test_batches = 1
        cfg.trainer.max_epochs = 1
        # for DEBUG purposes only

    fail_on_missing(cfg)
    pretty_print(cfg)
