# UNIREX: A Unified Learning Framework for Language Model Rationale Extraction (ICML 2022)

This is the official PyTorch repo for [UNIREX](https://arxiv.org/abs/2112.08802), a learning framework for jointly optimizing rationale extractors w.r.t. faithfulness, plausibility, and task performance.

```
UNIREX: A Unified Learning Framework for Language Model Rationale Extraction
Aaron Chan, Maziar Sanjabi, Lambert Mathias, Liang Tan, Shaoliang Nie, Xiaochang Peng, Xiang Ren, Hamed Firooz
ICML 2022
```

The majority of the UNIREX project is licensed under CC-BY-NC. However, some portions of the project are available under separate license terms, as indicated below:
- The [ERASER benchmark](https://github.com/jayded/eraserbenchmark) is licensed under the Apache License 2.0.

If UNIREX is helpful for your research, please consider citing our ICML paper:

```
@inproceedings{chan2022unirex,
  title={Unirex: A unified learning framework for language model rationale extraction},
  author={Chan, Aaron and Sanjabi, Maziar and Mathias, Lambert and Tan, Liang and Nie, Shaoliang and Peng, Xiaochang and Ren, Xiang and Firooz, Hamed},
  booktitle={International Conference on Machine Learning},
  pages={2867--2889},
  year={2022},
  organization={PMLR}
}
```

## Basics

### Neptune
Before running the code, you need to complete the following steps:
1. Create a [Neptune](https://neptune.ai/) account and project.
2. Edit the [project name](https://github.com/aarzchan/UNIREX/blob/main/configs/logger/neptune.yaml#L12), [local username](https://github.com/aarzchan/UNIREX/blob/main/src/utils/logging.py#L11), and [Neptune API token](https://github.com/aarzchan/UNIREX/blob/main/src/utils/logging.py#L11) fields in the code.


### Multirun
Do grid search over different configs.
```
python main.py -m \
    dataset=sst,stf \
    seed=0,1,2,3,4,5 \
```

### Evaluate checkpoint
This command evaluates a checkpoint on the train, dev, and test sets.
```
python main.py \
    training=evaluate \
    training.ckpt_path=/path/to/ckpt \
    training.eval_splits=train,dev,test \
```

### Finetune checkpoint
```
python main.py \
    training=evaluate \
    training.ckpt_path=/path/to/ckpt \
```

### Offline Mode
In offline mode, results are not logged to Neptune.
```
python main.py logger.offline=True
```

### Debug Mode
In debug mode, results are not logged to Neptune, and we only train/evaluate for limited number of batches and/or epochs.
```
python main.py debug=True
```

### Hydra Working Directory

Hydra will change the working directory to the path specified in `configs/hydra/default.yaml`. Therefore, if you save a file to the path `'./file.txt'`, it will actually save the file to somewhere like `logs/runs/xxxx/file.txt`. This is helpful when you want to version control your saved files, but not if you want to save to a global directory. There are two methods to get the "actual" working directory:

1. Use `hydra.utils.get_original_cwd` function call
2. Use `cfg.work_dir`. To use this in the config, can do something like `"${data_dir}/${.dataset}/${model.arch}/"`


### Config Key

- `work_dir` current working directory (where `src/` is)

- `data_dir` where data folder is

- `log_dir` where log folder is (runs & multirun)

- `root_dir` where the saved ckpt & hydra config are


---


## Example Commands

Here, we assume the following:
- The `data_dir` is `../data`, which means `data_dir=${work_dir}/../data`.
- The dataset is `sst`.

### 1. Build dataset
The commands below are used to build pre-processed datasets, saved as pickle files. The model architecture is specified so that we can use the correct tokenizer for pre-processing.

```
python scripts/build_dataset.py --data_dir ../data \
    --dataset sst --arch google/bigbird-roberta-base --split train

python scripts/build_dataset.py --data_dir ../data \
    --dataset sst --arch google/bigbird-roberta-base --split dev

python scripts/build_dataset.py --data_dir ../data \
    --dataset sst --arch google/bigbird-roberta-base --split test

```

If the dataset is very large, you have the option to subsample part of the dataset for smaller-scale experiements. For example, in the command below, we build a train set with only 1000 train examples (sampled with seed 0).
```
python scripts/build_dataset.py --data_dir ../data \
    --dataset sst --arch google/bigbird-roberta-base --split train \
    --num_train 1000 --num_train_seed 0
```

### 2. Train Task LM

The command below is the most basic way to run `main.py` and will train the Task LM without any explanation regularization (`model=lm`).

However, since all models need to be evaluated w.r.t. explainability metrics, we need to specify an attribution algorithm for computing post-hoc explanations. This is done by setting `model.explainer_type=attr_algo` to specify that we are using an attribution algorithm based explainer, `model.attr_algo` to specify the attribution algorithm, and `model.attr_pooling` to specify the attribution pooler.
```
python main.py -m \
    data=sst \
    model=lm \
    model.explainer_type=attr_algo \
    model.attr_algo=input-x-gradient \
    model.attr_pooling=sum \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

By default, checkpoints will not be saved, so you need to set `save_checkpoint=True` if you want to save the best checkpoint.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=lm \
    model.explainer_type=attr_algo \
    model.attr_algo=input-x-gradient \
    model.attr_pooling=sum \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

### 3. Train Task LM with explanation regularization
This repo implements a number of different methods for training the Task LM with explanation regularization. These methods aim to improve rationale faithfulness, plausibility, or both. Below are commands for running each method.


**Task LM + SGT**
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=attr_algo \
    model.attr_algo=input-x-gradient \
    model.attr_pooling=sum \
    model.task_wt=1.0 \
    model.comp_wt=0.0 \
    model.suff_wt=0.5 \
    model.suff_criterion=kldiv \
    model.plaus_wt=0.0 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + FRESH**

Train Task LM, saving the best checkpoint.

```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=lm \
    model.explainer_type=attr_algo \
    model.attr_algo=input-x-gradient \
    model.attr_pooling=sum \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0
```

Load Task LM checkpoint, then save its attributions to file.
Here, we assume the checkpoint path is based on a Neptune experiment ID with the form `ER-XXXX`, where `ER` is the Neptune project key.
```
python main.py -m \
    logger.offline=True \
    training=evaluate \
    training.ckpt_path="'ER-XXXX/checkpoints/epoch=X-step=XXXX.ckpt'" \
    training.eval_splits="'train,dev,test'" \
    data=sst \
    model=lm \
    model.explainer_type=attr_algo \
    model.attr_algo=input-x-gradient \
    model.attr_pooling=sum \
    model.save_outputs=True \
    model.exp_id="ER-XXXX" \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3
```

Load attributions, then retrain Task LM, using as input only the top-k% tokens w.r.t. the attributions.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    data.fresh_exp_id=ER-XXXX \
    data.fresh_attr_algo=input-x-gradient \
    data.fresh_topk=10 \
    model=fresh \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0
```

**Task LM + DLM (plaus)**

Use linear layer as DLM head.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=lm \
    model.expl_head_type=linear \
    model.task_wt=1.0 \
    model.comp_wt=0.0 \
    model.suff_wt=0.0 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

Use MLP as DLM head.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=lm \
    model.expl_head_type=mlp \
    model.expl_head_mlp_hidden_dim=2048 \
    model.expl_head_mlp_hidden_layers=2 \
    model.task_wt=1.0 \
    model.comp_wt=0.0 \
    model.suff_wt=0.0 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + SLM (plaus)**

Use linear layer as SLM head.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=self_lm \
    model.expl_head_type=linear \
    model.task_wt=1.0 \
    model.comp_wt=0.0 \
    model.suff_wt=0.0 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

Use MLP as SLM head.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=self_lm \
    model.expl_head_type=mlp \
    model.expl_head_mlp_hidden_dim=2048 \
    model.expl_head_mlp_hidden_layers=2 \
    model.task_wt=1.0 \
    model.comp_wt=0.0 \
    model.suff_wt=0.0 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + AA-Sum (comp/suff)**

Using Input*Grad attribution algorithm.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=attr_algo \
    model.attr_algo=input-x-gradient \
    model.attr_pooling=sum \
    model.task_wt=1.0 \
    model.comp_wt=0.5 \
    model.suff_wt=0.5 \
    model.plaus_wt=0.0 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

Using simple baseline for attribution algorithm.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=attr_algo \
    model.attr_algo={gold, inv, rand} \
    model.attr_pooling=sum \
    model.task_wt=1.0 \
    model.comp_wt=0.5 \
    model.suff_wt=0.5 \
    model.plaus_wt=0.0 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```


**Task LM + AA-Sum (comp/suff/plaus)**
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=attr_algo \
    model.attr_algo=input-x-gradient \
    model.attr_pooling=sum \
    model.task_wt=1.0 \
    model.comp_wt=0.5 \
    model.suff_wt=0.5 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

### **Task LM + AA-MLP (comp/suff/plaus)**
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=attr_algo \
    model.attr_algo=input-x-gradient \
    model.attr_pooling=mlp \
    model.attr_mlp_hidden_dim=2048 \
    model.attr_mlp_hidden_layers=2 \
    model.task_wt=1.0 \
    model.comp_wt=0.5 \
    model.suff_wt=0.5 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + DLM (comp/suff/plaus)**
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=lm \
    model.expl_head_type=linear \
    model.task_wt=1.0 \
    model.comp_wt=0.5 \
    model.suff_wt=0.5 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

Use MLP as DLM head.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=lm \
    model.expl_head_type=mlp \
    model.expl_head_mlp_hidden_dim=2048 \
    model.expl_head_mlp_hidden_layers=2 \
    model.task_wt=1.0 \
    model.comp_wt=0.5 \
    model.suff_wt=0.5 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + SLM (comp/suff/plaus)**
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=self_lm \
    model.expl_head_type=linear \
    model.task_wt=1.0 \
    model.comp_wt=0.5 \
    model.suff_wt=0.5 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

Use MLP as SLM head.
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=expl_reg \
    model.explainer_type=self_lm \
    model.expl_head_type=mlp \
    model.expl_head_mlp_hidden_dim=2048 \
    model.expl_head_mlp_hidden_layers=2 \
    model.task_wt=1.0 \
    model.comp_wt=0.5 \
    model.suff_wt=0.5 \
    model.plaus_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + L2E**
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    data.l2e_exp_id=ER-XXXX \
    data.l2e_attr_algo=integrated-gradients \
    model=expl_reg \
    model.explainer_type=lm \
    model.expl_head_type=linear \
    model.task_wt=1.0 \
    model.plaus_wt=0.5 \
    model.l2e=True \
    model.l2e_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Task LM + A2R**
```
python main.py -m \
    save_checkpoint=True \
    data=sst \
    model=a2r \
    model.task_wt=1.0 \
    model.plaus_wt=0.5 \
    model.a2r_wt=0.5 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=32 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=32 \
    setup.eval_batch_size=32 \
    setup.num_workers=3 \
    seed=0,1,2
```

**Measure Explainer Runtime**
```
python main.py -m \
    training=evaluate \
    training.ckpt_path="'save/ER-XXX/checkpoints/epoch=X-step=X.ckpt'" \
    data=sst \
    model=lm \
    model.explainer_type=attr_algo \
    model.attr_algo=integrated-gradients \
    model.ig_steps=3 \
    model.return_convergence_delta=True \
    model.internal_batch_size=3 \
    model.attr_pooling=sum \
    model.measure_attrs_runtime=True \
    setup.train_batch_size=1 \
    setup.eff_train_batch_size=1 \
    setup.eval_batch_size=1 \
    setup.num_workers=3 \
    seed=0,1,2
```
