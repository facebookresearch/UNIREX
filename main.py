import time

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # import here for faster auto completion
    from src.utils.conf import touch
    from src.run import run

    # additional set field by condition
    # assert no missing etc
    touch(cfg)

    start_time = time.time()
    metric = run(cfg)
    print(
        f'Time Taken for experiment {cfg.logger.neptune_exp_id}: {(time.time() - start_time) / 3600}h')

    return metric


if __name__ == '__main__':
    main()
