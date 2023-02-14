import os
import types
from pathlib import Path
from typing import Optional
from copy import deepcopy
from itertools import chain

import numpy as np
import pickle5 as pickle
from hydra.utils import get_original_cwd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.data import dataset_info, data_keys


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset: str, data_path: str, mode: str,
                 train_batch_size: int = 1, eval_batch_size: int = 1, eff_train_batch_size: int = 1, num_workers: int = 0,
                 num_train: int = None, num_dev: int = None, num_test: int = None,
                 num_train_seed: int = None, num_dev_seed: int = None, num_test_seed: int = None,
                 pct_train_rationales: float = None, pct_train_rationales_seed: int = None, train_rationales_batch_factor: float = None,
                 neg_weight: float = 1,
                 attr_algo: str = None,
                 fresh_exp_id: str = None, fresh_attr_algo: str = None, fresh_topk: int = None,
                 fresh_extractor = None, min_val: float = -1e10,
                 l2e_exp_id: str = None, l2e_attr_algo: str = None, l2e_num_classes: int = 5,
                 train_shuffle: bool = False,
                 ):
        super().__init__()

        self.dataset = dataset
        self.data_path = data_path # ${data_dir}/${.dataset}/${model.arch}/

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.eff_train_batch_size = eff_train_batch_size
        self.num_workers = num_workers

        self.num_samples = {'train': num_train, 'dev': num_dev, 'test': num_test}
        self.num_samples_seed = {'train': num_train_seed, 'dev': num_dev_seed, 'test': num_test_seed}
        self.pct_train_rationales = pct_train_rationales
        self.pct_train_rationales_seed = pct_train_rationales_seed
        self.train_rationales_batch_factor = train_rationales_batch_factor

        self.attr_algo = attr_algo

        self.fresh_exp_id = fresh_exp_id
        self.fresh_attr_algo = fresh_attr_algo
        self.fresh_topk = fresh_topk
        if fresh_topk is not None:
            assert 0 < fresh_topk < 100
        self.fresh_extractor = fresh_extractor
        self.min_val = min_val

        self.l2e_exp_id = l2e_exp_id
        self.l2e_attr_algo = l2e_attr_algo
        self.l2e_num_classes = l2e_num_classes

        self.train_shuffle = train_shuffle

    def load_dataset(self, split):
        dataset = {}
        data_path = os.path.join(self.data_path, split)
        assert Path(data_path).exists()
        
        for key in tqdm(data_keys, desc=f'Loading {split} set'):
            if (
                (key in ['inv_rationale', 'rand_rationale'] and self.attr_algo not in ['inv', 'rand'])
                or (key == 'rationale' and self.dataset in ['amazon', 'yelp', 'stf', 'olid', 'irony'])
                or (key == 'rationale_indices' and (split != 'train' or self.pct_train_rationales is None))
            ):
                continue
            elif key == 'rationale_indices' and split == 'train' and self.pct_train_rationales is not None:
                filename = f'{key}_{self.pct_train_rationales}_{self.pct_train_rationales_seed}.pkl'
            elif self.num_samples[split] is not None:
                filename = f'{key}_{self.num_samples[split]}_{self.num_samples_seed[split]}.pkl'
            else:
                filename = f'{key}.pkl'

            with open(os.path.join(data_path, filename), 'rb') as f:
                dataset[key] = pickle.load(f)

        if self.fresh_exp_id is not None:
            assert self.fresh_attr_algo in ['integrated-gradients', 'input-x-gradient', 'lm-gold']
            assert self.fresh_extractor is not None
            assert self.fresh_topk is not None

            fresh_rationale_path = f'{get_original_cwd()}/../save/{self.fresh_exp_id}/model_outputs/{self.dataset}/{split}_attrs.pkl'
            print(f'Using {fresh_rationale_path} for split: {split}\n')
            assert Path(fresh_rationale_path).exists()
            attrs = pickle.load(open(fresh_rationale_path, 'rb'))
            if self.dataset == 'cose':
                num_examples = dataset_info[self.dataset][split][1]
                num_classes = dataset_info[self.dataset]['num_classes']
                attrs = attrs.reshape(num_examples, num_classes, -1)

            fresh_rationales = []
            for i, attr in enumerate(attrs):
                attn_mask = torch.LongTensor(dataset['attention_mask'][i])
                if self.dataset == 'cose':
                    fresh_rationale = []
                    for j, a in enumerate(attn_mask):
                        num_tokens = a.sum() - 1 # don't include CLS token when computing num_tokens
                        cur_attr = attr[j] + (1 - a) * self.min_val # ignore pad tokens when computing topk_indices
                        cur_attr[0] = self.min_val # don't include CLS token when computing topk_indices
                        k = max(1, int(self.fresh_topk / 100 * num_tokens))
                        cur_fresh_rationale = torch.zeros_like(a).long()
                        topk_indices = torch.argsort(cur_attr[:num_tokens], descending=True)[:k]
                        cur_fresh_rationale[topk_indices] = 1
                        cur_fresh_rationale[0] = 1 # always treat CLS token as positive token
                        fresh_rationale.append(list(cur_fresh_rationale.numpy()))
                else:
                    num_tokens = attn_mask.sum() - 1 # don't include CLS token when computing num_tokens
                    cur_attr = attr + (1 - attn_mask) * self.min_val # ignore pad tokens when computing topk_indices
                    cur_attr[0] = self.min_val # don't include CLS token when computing topk_indices
                    k = max(1, int(self.fresh_topk / 100 * num_tokens))
                    fresh_rationale = torch.zeros_like(attn_mask).long()
                    topk_indices = torch.argsort(cur_attr[:num_tokens], descending=True)[:k]
                    fresh_rationale[topk_indices] = 1
                    fresh_rationale[0] = 1 # always treat CLS token as positive token
                    fresh_rationale = list(fresh_rationale.numpy())

                fresh_rationales.append(fresh_rationale)

            if self.fresh_extractor == 'oracle':
                dataset['fresh_rationale'] = fresh_rationales
            else:
                raise NotImplementedError

        elif self.l2e_exp_id is not None:
            assert self.l2e_attr_algo in ['integrated-gradients']
            l2e_rationale_path = f'{get_original_cwd()}/../save/{self.l2e_exp_id}/model_outputs/{self.dataset}/{split}_attrs.pkl'
            print(f'Using {l2e_rationale_path} for split: {split}\n')
            assert Path(l2e_rationale_path).exists()
            attrs = pickle.load(open(l2e_rationale_path, 'rb'))
            if self.dataset == 'cose':
                num_examples = dataset_info[self.dataset][split][1]
                num_classes = dataset_info[self.dataset]['num_classes']
                attrs = attrs.reshape(num_examples, num_classes, -1)
            l2e_rationales = [list(attr.numpy()) for attr in attrs]
            dataset['l2e_rationale'] = l2e_rationales

        if split == 'train' and self.pct_train_rationales is not None:
            dataset_ = deepcopy(dataset)
            dataset_keys = dataset_.keys()
            rationale_indices = dataset_['rationale_indices']
            dataset, train_rationales_dataset = {}, {}
            for key in dataset_keys:
                if key != 'rationale_indices':
                    dataset[key] = [x for i, x in enumerate(dataset_[key]) if i not in rationale_indices]
                    train_rationales_dataset[key] = [x for i, x in enumerate(dataset_[key]) if i in rationale_indices]
            assert sorted(rationale_indices) == train_rationales_dataset['item_idx']
        else:
            train_rationales_dataset = None

        return dataset, train_rationales_dataset

    def setup(self, splits=['all']):
        self.data = {}
        splits = ['train', 'dev', 'test'] if splits == ['all'] else splits
        for split in splits:
            dataset, train_rationales_dataset = self.load_dataset(split)
            self.data[split] = TextClassificationDataset(dataset, split, train_rationales_dataset, self.train_batch_size, self.l2e_num_classes)

    def train_dataloader(self):
        if self.pct_train_rationales is not None:
            assert self.train_batch_size >= 2
            assert self.train_rationales_batch_factor > 1
            batch_size = self.train_batch_size - int(max(1, self.train_batch_size / self.train_rationales_batch_factor))
        else:
            batch_size = self.train_batch_size

        return DataLoader(
            self.data['train'],
            batch_size=batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data['train'].collater,
            shuffle=self.train_shuffle,
            pin_memory=True
        )

    def val_dataloader(self, test=False):
        if test:
            return DataLoader(
                self.data['dev'],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data['dev'].collater,
                pin_memory=True
            )

        return [
            DataLoader(
            self.data[eval_split],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data[eval_split].collater,
            pin_memory=True)
            
            for eval_split in ['dev', 'test']
        ]

    def test_dataloader(self):
        return DataLoader(
            self.data['test'],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data['test'].collater,
            pin_memory=True
        )


class TextClassificationDataset(Dataset):
    def __init__(self, dataset, split, train_rationales_dataset=None, train_batch_size=None, l2e_num_classes=None):
        self.data = dataset
        self.split = split
        self.train_rationales_dataset = train_rationales_dataset
        self.train_batch_size = train_batch_size
        assert not (split != 'train' and train_rationales_dataset is not None)
        if train_rationales_dataset is not None:
            self.len_train_rationales_dataset = len(train_rationales_dataset['item_idx'])
        self.l2e_num_classes = l2e_num_classes

    def __len__(self):
        return len(self.data['item_idx'])

    def __getitem__(self, idx):
        item_idx = torch.LongTensor([self.data['item_idx'][idx]])
        input_ids = torch.LongTensor(self.data['input_ids'][idx])
        attention_mask = torch.LongTensor(self.data['attention_mask'][idx])
        rationale = torch.FloatTensor(self.data['rationale'][idx]) if self.data.get('rationale') else None
        has_rationale = torch.LongTensor([self.data['has_rationale'][idx]])
        if self.train_rationales_dataset is not None:
            has_rationale *= 0
        label = torch.LongTensor([self.data['label'][idx]])
        inv_rationale = torch.FloatTensor(self.data['inv_rationale'][idx]) if self.data.get('inv_rationale') else None
        rand_rationale = torch.FloatTensor(self.data['rand_rationale'][idx]) if self.data.get('rand_rationale') else None
        fresh_rationale = torch.LongTensor(self.data['fresh_rationale'][idx]) if self.data.get('fresh_rationale') else None
        if self.data.get('l2e_rationale'):
            assert self.l2e_num_classes == 5
            l2e_rationale = self.discretize_l2e_rationale(self.data['l2e_rationale'][idx])
        else:
            l2e_rationale = None

        return (
            item_idx, input_ids, attention_mask, rationale, has_rationale, label, inv_rationale, rand_rationale, fresh_rationale, l2e_rationale
        )

    def discretize_l2e_rationale(self, l2e_rationale):
        l2e_rationale = list(np.array(l2e_rationale).flatten())

        all_pos = [x for x in l2e_rationale if x > 0]
        all_neg = [x for x in l2e_rationale if x < 0]
        mean_pos = sum(all_pos) / len(all_pos) if len(all_pos) > 0 else 0.0
        mean_neg = sum(all_neg) / len(all_neg) if len(all_neg) > 0 else 0.0
        
        l2e_rationale = torch.LongTensor([
            0 if x < mean_neg
            else 1 if mean_neg <= x < 0.0
            else 2 if x == 0.0
            else 3 if 0.0 < x <= mean_pos 
            else 4
        for x in l2e_rationale])
        
        return l2e_rationale

    def sample_train_rationale_indices(self, num_samples):
        return list(np.random.choice(self.len_train_rationales_dataset, size=num_samples, replace=False))

    def get_train_rationale_item(self, idx):
        item_idx = torch.LongTensor([self.train_rationales_dataset['item_idx'][idx]])
        input_ids = torch.LongTensor(self.train_rationales_dataset['input_ids'][idx])
        attention_mask = torch.LongTensor(self.train_rationales_dataset['attention_mask'][idx])
        rationale = torch.FloatTensor(self.train_rationales_dataset['rationale'][idx])
        has_rationale = torch.LongTensor([self.train_rationales_dataset['has_rationale'][idx]])
        label = torch.LongTensor([self.train_rationales_dataset['label'][idx]])
        inv_rationale = torch.FloatTensor(self.train_rationales_dataset['inv_rationale'][idx]) if self.train_rationales_dataset.get('inv_rationale') else None
        rand_rationale = torch.FloatTensor(self.train_rationales_dataset['rand_rationale'][idx]) if self.train_rationales_dataset.get('rand_rationale') else None
        fresh_rationale = torch.LongTensor(self.train_rationales_dataset['fresh_rationale'][idx]) if self.train_rationales_dataset.get('fresh_rationale') else None
        if self.train_rationales_dataset.get('l2e_rationale'):
            assert self.l2e_num_classes == 5
            l2e_rationale = self.discretize_l2e_rationale(self.train_rationales_dataset['l2e_rationale'][idx])
        else:
            l2e_rationale = None

        return (
            item_idx, input_ids, attention_mask, rationale, has_rationale, label, inv_rationale, rand_rationale, fresh_rationale, l2e_rationale
        )

    def collater(self, items):
        batch_size = len(items)
        if self.train_rationales_dataset is not None:
            num_train_rationale_indices = int(max(1, self.train_batch_size - batch_size))
            train_rationale_indices = self.sample_train_rationale_indices(num_train_rationale_indices)
            for idx in train_rationale_indices:
                items.append(self.get_train_rationale_item(idx))

        batch = {
            'item_idx': torch.cat([x[0] for x in items]),
            'input_ids': torch.stack([x[1] for x in items], dim=0),
            'attention_mask': torch.stack([x[2] for x in items], dim=0),
            'rationale': torch.stack([x[3] for x in items], dim=0) if self.data.get('rationale') else None,
            'has_rationale': torch.cat([x[4] for x in items]),
            'label': torch.cat([x[5] for x in items]),
            'inv_rationale': torch.stack([x[6] for x in items], dim=0) if self.data.get('inv_rationale') else None,
            'rand_rationale': torch.stack([x[7] for x in items], dim=0) if self.data.get('rand_rationale') else None,
            'fresh_rationale': torch.stack([x[8] for x in items], dim=0) if self.data.get('fresh_rationale') else None,
            'l2e_rationale': torch.stack([x[9] for x in items], dim=0) if self.data.get('l2e_rationale') else None,
            'split': self.split, # when evaluate_ckpt=true, split always test
        }
        
        return batch