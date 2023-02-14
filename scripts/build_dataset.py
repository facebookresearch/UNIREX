import argparse, json, math, os, sys, random, logging
from collections import defaultdict as ddict, Counter
from itertools import chain

import numpy as np
import pandas as pd
from pickle5 import pickle
from tqdm import tqdm

import torch
import datasets
from transformers import AutoTokenizer

sys.path.append(os.path.join(sys.path[0], '..'))
from src.utils.data import dataset_info, eraser_datasets, data_keys
from src.utils.eraser.utils import annotations_from_jsonl, load_documents
from src.utils.eraser.data_utils import (
    bert_tokenize_doc,
    bert_intern_doc,
    bert_intern_annotation,
    annotations_to_evidence_identification,
    annotations_to_evidence_token_identification,
)

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_dataset_dict(
    idx, dataset_dict, input_ids, rationale, max_length, actual_max_length, tokenizer, interned_annotations, classes
):
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    rationale = [0] + rationale + [0]
    assert len(input_ids) == len(rationale)
    num_tokens = len(input_ids)
    if num_tokens > actual_max_length:
        actual_max_length = num_tokens

    num_pad_tokens = max_length - num_tokens
    assert num_pad_tokens >= 0

    input_ids += [tokenizer.pad_token_id] * num_pad_tokens
    attention_mask = [1] * num_tokens + [0] * num_pad_tokens
    rationale += [0] * num_pad_tokens

    inv_rationale = [1.0-x for x in rationale]
    rand_rationale = list(np.random.randn(max_length))

    has_rationale = int(sum(rationale) > 0)
    if has_rationale == 0:
        raise ValueError('empty rationale')

    label = classes.index(interned_annotations[idx].classification)

    dataset_dict['item_idx'].append(idx)
    dataset_dict['input_ids'].append(input_ids)
    dataset_dict['attention_mask'].append(attention_mask)
    dataset_dict['rationale'].append(rationale)
    dataset_dict['inv_rationale'].append(inv_rationale)
    dataset_dict['rand_rationale'].append(rand_rationale)
    dataset_dict['has_rationale'].append(has_rationale)
    dataset_dict['label'].append(label)

    return dataset_dict, actual_max_length

def align_rationale_with_tokens(input_ids, raw_tokens, raw_rationale, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    rationale = []
    j = 0
    cur_token = tokens[j]

    for i in range(len(raw_tokens)):
        cur_raw_token = raw_tokens[i]
        cur_raw_rationale = raw_rationale[i]
        cur_reconstructed_raw_token = ''

        while len(cur_raw_token) > 0:
            for char in cur_token:
                if char == cur_raw_token[0]:
                    cur_raw_token = cur_raw_token[1:]
                    cur_reconstructed_raw_token += char

            rationale.append(cur_raw_rationale)
            j += 1
            cur_token = tokens[j] if j < len(tokens) else None

        assert cur_reconstructed_raw_token == raw_tokens[i]

    return rationale

def stratified_sampling(data, num_samples):
    num_instances = len(data)
    assert num_samples < num_instances

    counter_dict = Counter(data)
    unique_vals = list(counter_dict.keys())
    val_counts = list(counter_dict.values())
    num_unique_vals = len(unique_vals)
    assert num_unique_vals > 1

    num_stratified_samples = [int(c*num_samples/num_instances) for c in val_counts]
    assert sum(num_stratified_samples) <= num_samples
    if sum(num_stratified_samples) < num_samples:
        delta = num_samples - sum(num_stratified_samples)
        delta_samples = np.random.choice(range(num_unique_vals), replace=True, size=delta)
        for val in delta_samples:
            num_stratified_samples[unique_vals.index(val)] += 1
    assert sum(num_stratified_samples) == num_samples

    sampled_indices = []
    for i, val in enumerate(unique_vals):
        candidates = np.where(data == val)[0]
        sampled_indices += list(np.random.choice(candidates, replace=False, size=num_stratified_samples[i]))
    random.shuffle(sampled_indices)

    return sampled_indices

def sample_dataset(data_path, dataset_dict, split, num_samples, seed):
    sampled_split_filename = f'{split}_split_{num_samples}_{seed}.pkl'
    if os.path.exists(os.path.join(data_path, sampled_split_filename)):
        with open(os.path.join(data_path, sampled_split_filename), 'rb') as f:
            sampled_split = pickle.load(f)
    else:
        sampled_split = stratified_sampling(dataset_dict['label'], num_samples)
        with open(os.path.join(data_path, sampled_split_filename), 'wb') as f:
            pickle.dump(sampled_split, f)

    for key in data_keys:
        dataset_dict[key] = sampled_split if key == 'item_idx' else [dataset_dict[key][i] for i in sampled_split]

    return dataset_dict

def sample_rationale_indices(data_path, dataset_dict, num_examples, pct_train_rationales, seed):
    # Sample indices for train examples with gold rationales
    sampled_indices_filename = f'rationale_indices_{pct_train_rationales}_{seed}.pkl'
    if not os.path.exists(os.path.join(data_path, sampled_indices_filename)):
        num_train_rationales = int(math.ceil(num_examples * pct_train_rationales / 100))
        sampled_indices = list(np.random.choice(dataset_dict['item_idx'], size=num_train_rationales, replace=False))
        with open(os.path.join(data_path, sampled_indices_filename), 'wb') as f:
            pickle.dump(sampled_indices, f)

def load_dataset(data_path):
    dataset_dict = ddict(list)
    for key in tqdm(data_keys, desc=f'Loading {args.split} dataset'):
        with open(os.path.join(data_path, f'{key}.pkl'), 'rb') as f:
            dataset_dict[key] = pickle.load(f)
    return dataset_dict

def save_dataset(data_path, dataset_dict, split, num_samples, seed):
    for key in tqdm(data_keys, desc=f'Saving {split} dataset'):
        filename = f'{key}.pkl' if num_samples is None else f'{key}_{num_samples}_{seed}.pkl'
        with open(os.path.join(data_path, filename), 'wb') as f:
            pickle.dump(dataset_dict[key], f)

def main(args):
    set_random_seed(args.seed)

    assert args.split is not None and args.arch is not None
    assert args.num_samples is None or args.num_samples >= 1

    split, num_examples = dataset_info[args.dataset][args.split]
    if args.num_samples is not None:
        assert args.num_samples < num_examples

    num_classes = dataset_info[args.dataset]['num_classes']
    max_length = dataset_info[args.dataset]['max_length'][args.arch]
    num_special_tokens = dataset_info[args.dataset]['num_special_tokens']
    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    data_path = os.path.join(args.data_dir, args.dataset, args.arch, args.split)
    classes = dataset_info[args.dataset]['classes']
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if args.dataset in eraser_datasets:
        eraser_path = os.path.join(args.data_dir, 'eraser', args.dataset)
        documents_path = os.path.join(args.data_dir, args.dataset, args.arch, 'documents.pkl')
        documents = load_documents(eraser_path)
        logger.info(f'Loaded {len(documents)} documents')
        if os.path.exists(documents_path):
            logger.info(f'Loading processed documents from {documents_path}')
            (interned_documents, interned_document_token_slices) = torch.load(documents_path)
            logger.info(f'Loaded {len(interned_documents)} processed documents')
        else:
            logger.info(f'Processing documents')
            special_token_map = {
                                    'SEP': [tokenizer.sep_token_id],
                                    '[SEP]': [tokenizer.sep_token_id],
                                    '[sep]': [tokenizer.sep_token_id],
                                    'UNK': [tokenizer.unk_token_id],
                                    '[UNK]': [tokenizer.unk_token_id],
                                    '[unk]': [tokenizer.unk_token_id],
                                    'PAD': [tokenizer.unk_token_id],
                                    '[PAD]': [tokenizer.unk_token_id],
                                    '[pad]': [tokenizer.unk_token_id],
                                }
            interned_documents = {}
            interned_document_token_slices = {}
            for d, doc in tqdm(documents.items(), desc='Processing documents'):
                tokenized, w_slices = bert_tokenize_doc(doc, tokenizer, special_token_map=special_token_map)
                interned_documents[d] = bert_intern_doc(tokenized, tokenizer, special_token_map=special_token_map)
                interned_document_token_slices[d] = w_slices
            logger.info(f'Saving processed documents to {documents_path}')
            torch.save((interned_documents, interned_document_token_slices), documents_path)
            sys.exit()

        annotations_path = os.path.join(eraser_path, f'{split}.jsonl')
        annotations = annotations_from_jsonl(annotations_path)
        interned_annotations = bert_intern_annotation(annotations, tokenizer)
        if args.dataset in ['cose', 'esnli', 'movies']:
            evidence_data = annotations_to_evidence_token_identification(annotations, documents, interned_documents, interned_document_token_slices)
        elif args.dataset in ['fever', 'multirc']:
            evidence_data = annotations_to_evidence_identification(annotations, interned_documents)
        assert len(evidence_data) == num_examples

    missing_data_keys = [x for x in data_keys if not os.path.exists(os.path.join(data_path, f'{x}.pkl'))]
    if args.num_samples is None and missing_data_keys:
        dataset_dict = ddict(list)
        actual_max_length = 0
        if args.dataset in eraser_datasets:
            if args.dataset not in ['cose', 'esnli', 'fever', 'movies', 'multirc']:
                raise NotImplementedError

            if args.dataset == 'cose':
                q_marker = tokenizer('Q:', add_special_tokens=False)['input_ids']
                a_marker = tokenizer('A:', add_special_tokens=False)['input_ids']
                for idx, (instance_id, instance_evidence) in tqdm(enumerate(evidence_data.items()), desc=f'Building {args.split} dataset', total=num_examples):
                    instance_docs = ddict(dict)
                    assert len(instance_evidence) == 1
                    doc = interned_documents[instance_id]
                    evidence_sentences = instance_evidence[instance_id]

                    question = list(chain.from_iterable(doc))
                    question_rationale = list(chain.from_iterable([x.kls for x in evidence_sentences]))
                    answers = evidence_sentences[0].query.split(' [sep] ')
                    answer_ids = [tokenizer(x, add_special_tokens=False)['input_ids'] for x in answers]

                    input_ids, attention_mask, rationale, inv_rationale, rand_rationale, has_rationale = [], [], [], [], [], []
                    for answer in answer_ids:
                        cur_input_ids = [tokenizer.cls_token_id] + q_marker + question + [tokenizer.sep_token_id] + a_marker + answer + [tokenizer.sep_token_id]

                        num_tokens = len(cur_input_ids)
                        if num_tokens > actual_max_length:
                            actual_max_length = num_tokens
                        num_pad_tokens = max_length - num_tokens
                        assert num_pad_tokens >= 0

                        cur_input_ids += [tokenizer.pad_token_id] * num_pad_tokens
                        input_ids.append(cur_input_ids)

                        cur_attention_mask = [1] * num_tokens + [0] * num_pad_tokens
                        attention_mask.append(cur_attention_mask)

                        cur_rationale = [0] + [0]*len(q_marker) + question_rationale + [0] + [0]*len(a_marker) + [0]*len(answer) + [0]
                        cur_rationale += [0] * num_pad_tokens
                        assert len(cur_input_ids) == len(cur_rationale)
                        rationale.append(cur_rationale)

                        inv_rationale.append([1.0-x for x in cur_rationale])
                        rand_rationale.append(list(np.random.randn(max_length)))

                        cur_has_rationale = int(sum(cur_rationale) > 0)
                        if cur_has_rationale == 0:
                            raise ValueError('empty rationale')
                        has_rationale.append(cur_has_rationale)

                    label = classes.index(interned_annotations[idx].classification)

                    dataset_dict['item_idx'].append(idx)
                    dataset_dict['input_ids'].append(input_ids)
                    dataset_dict['attention_mask'].append(attention_mask)
                    dataset_dict['rationale'].append(rationale)
                    dataset_dict['inv_rationale'].append(inv_rationale)
                    dataset_dict['rand_rationale'].append(rand_rationale)
                    dataset_dict['has_rationale'].append(has_rationale)
                    dataset_dict['label'].append(label)

            elif args.dataset == 'esnli':
                for idx, (instance_id, instance_evidence) in tqdm(enumerate(evidence_data.items()), desc=f'Building {args.split} dataset', total=num_examples):
                    instance_docs = ddict(dict)
                    assert len(instance_evidence) in [1, 2]
                    for doc_type in ['premise', 'hypothesis']:
                        doc_id = f'{instance_id}_{doc_type}'
                        doc = instance_evidence[doc_id]
                        if doc:
                            instance_docs[doc_type]['text'] = doc[0][5]
                            instance_docs[doc_type]['rationale'] = list(doc[0][0])
                        else:
                            instance_docs[doc_type]['text'] = interned_documents[doc_id][0]
                            instance_docs[doc_type]['rationale'] = [0] * len(interned_documents[doc_id][0])

                    input_ids = instance_docs['premise']['text'] + [tokenizer.sep_token_id] + instance_docs['hypothesis']['text']
                    assert all([x != tokenizer.unk_token_id for x in input_ids])
                    rationale = instance_docs['premise']['rationale'] + [0] + instance_docs['hypothesis']['rationale']
                    dataset_dict, actual_max_length = update_dataset_dict(idx, dataset_dict, input_ids, rationale, max_length, actual_max_length, tokenizer, interned_annotations, classes)

            elif args.dataset == 'movies':
                for idx, (instance_id, instance_evidence) in tqdm(enumerate(evidence_data.items()), desc=f'Building {args.split} dataset', total=num_examples):
                    instance_docs = ddict(dict)
                    assert len(instance_evidence) == 1
                    doc_id = list(instance_evidence.keys())[0]
                    doc = interned_documents[doc_id]
                    evidence_sentences = instance_evidence[doc_id]

                    input_ids = list(chain.from_iterable(doc))
                    rationale = list(chain.from_iterable([x.kls for x in evidence_sentences]))

                    input_length = min(len(input_ids), max_length - num_special_tokens)
                    if sum(rationale[:input_length]) > 0:
                        input_ids = input_ids[:input_length]
                        rationale = rationale[:input_length]
                    else:
                        input_ids = input_ids[-input_length:]
                        rationale = rationale[-input_length:]

                    dataset_dict, actual_max_length = update_dataset_dict(idx, dataset_dict, input_ids, rationale, max_length, actual_max_length, tokenizer, interned_annotations, classes)

            elif args.dataset == 'multirc':
                for idx, (instance_id, instance_evidence) in tqdm(enumerate(evidence_data.items()), desc=f'Building {args.split} dataset', total=num_examples):
                    instance_docs = ddict(dict)
                    assert len(instance_evidence) == 1
                    doc_id = list(instance_evidence.keys())[0]
                    doc = interned_documents[doc_id]
                    evidence_sentences = [x for x in instance_evidence[doc_id] if x.kls == 1]
                    evidence_indices = [x.index for x in evidence_sentences]

                    evidence_ids = list(chain.from_iterable(doc))
                    query_ids = tokenizer(instance_evidence[doc_id][0].query, add_special_tokens=False)['input_ids']
                    rationale = []
                    for i, sentence in enumerate(doc):
                        if i in evidence_indices:
                            rationale += [1] * len(sentence)
                        else:
                            rationale += [0] * len(sentence)

                    input_ids = evidence_ids + [tokenizer.sep_token_id] + query_ids
                    rationale += [0] * (len(query_ids)+1)
                    dataset_dict, actual_max_length = update_dataset_dict(idx, dataset_dict, input_ids, rationale, max_length, actual_max_length, tokenizer, interned_annotations, classes)

            else:
                raise NotImplementedError

        elif args.dataset == 'sst':
            sst_json_path = open(os.path.join(args.data_dir, args.dataset, f'sst_{split}.json'))
            dataset = json.load(sst_json_path)
            for idx in tqdm(range(num_examples), desc=f'Building {args.split} dataset'):
                instance = dataset[idx]

                text = instance['text']
                raw_tokens = text.split()
                raw_rationale = [1.0 if x >= 0.5 else 0.0 for x in instance['rationale']]
                assert len(raw_tokens) >= len(raw_rationale)
                if len(raw_tokens) > len(raw_rationale): # rationale missing last token for train instances [3631, 4767, 6020]
                    diff = len(raw_tokens) - len(raw_rationale) # diff always equals 1
                    raw_rationale = raw_rationale + diff*[0.0]
                assert len(raw_tokens) == len(raw_rationale)
                assert len(raw_tokens) <= max_length

                input_ids = tokenizer(text, add_special_tokens=False)['input_ids']
                rationale = align_rationale_with_tokens(input_ids, raw_tokens, raw_rationale, tokenizer)
                assert len(input_ids) == len(rationale)

                num_tokens = len(input_ids) + num_special_tokens
                if num_tokens > actual_max_length:
                    actual_max_length = num_tokens
                num_pad_tokens = max_length - num_tokens
                assert num_pad_tokens >= 0

                input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id] + num_pad_tokens*[tokenizer.pad_token_id]
                assert all([x != tokenizer.unk_token_id for x in input_ids])
                assert len(input_ids) == max_length

                attn_mask = num_tokens*[1] + num_pad_tokens*[0]
                assert len(attn_mask) == max_length

                rationale = [0.0] + rationale + [0.0] + num_pad_tokens*[0.0]
                assert sum(rationale) > 0
                assert len(rationale) == max_length

                inv_rationale = [1.0-x for x in rationale]
                rand_rationale = list(np.random.randn(max_length))

                label = classes.index(instance['classification'])

                dataset_dict['item_idx'].append(idx)
                dataset_dict['input_ids'].append(input_ids)
                dataset_dict['attention_mask'].append(attn_mask)
                dataset_dict['rationale'].append(rationale)
                dataset_dict['inv_rationale'].append(inv_rationale)
                dataset_dict['rand_rationale'].append(rand_rationale)
                dataset_dict['has_rationale'].append(1)
                dataset_dict['label'].append(label)

        elif args.dataset == 'stf':
            stf_raw_path = os.path.join(args.data_dir, args.dataset, 'stf_raw', f'{split}.tsv')
            dataset = pd.read_csv(stf_raw_path, sep='\t')
            for row in tqdm(dataset.itertuples(index=True), desc=f'Building {args.split} dataset'):
                idx, _, text, label = row
                tokens = tokenizer(text, padding='max_length', max_length=max_length, truncation=True)

                input_ids_ = tokenizer(text, add_special_tokens=False)['input_ids']
                num_tokens = len(input_ids_) + num_special_tokens
                if num_tokens > actual_max_length:
                    actual_max_length = num_tokens

                rand_rationale = list(np.random.randn(max_length))

                dataset_dict['item_idx'].append(idx)
                dataset_dict['input_ids'].append(tokens['input_ids'])
                dataset_dict['attention_mask'].append(tokens['attention_mask'])
                dataset_dict['rationale'].append(None)
                dataset_dict['rand_rationale'].append(rand_rationale)
                dataset_dict['has_rationale'].append(0)
                dataset_dict['label'].append(label)

        elif args.dataset == 'yelp':
            split_ = 'train' if split != 'test' else 'test'
            dataset = datasets.load_dataset('yelp_polarity')[split_]

            if split in ['train', 'test']:
                start_idx = 0
            elif split == 'dev':
                start_idx = dataset_info[args.dataset][split_][1]

            for idx in tqdm(range(start_idx, start_idx+num_examples), desc=f'Building {args.split} dataset'):
                text = dataset[idx]['text']
                tokens = tokenizer(
                    text,
                    padding='max_length',
                    max_length=max_length,
                    truncation=True
                )

                input_ids_ = tokenizer(text, add_special_tokens=False)['input_ids']
                num_tokens = len(input_ids_) + num_special_tokens
                if num_tokens > actual_max_length:
                    actual_max_length = num_tokens

                rand_rationale = list(np.random.randn(max_length))

                dataset_dict['item_idx'].append(idx-start_idx)
                dataset_dict['input_ids'].append(tokens['input_ids'])
                dataset_dict['attention_mask'].append(tokens['attention_mask'])
                dataset_dict['rationale'].append(None)
                dataset_dict['rand_rationale'].append(rand_rationale)
                dataset_dict['has_rationale'].append(0)
                dataset_dict['label'].append(dataset[idx]['label'])

        elif args.dataset == 'amazon':
            split_ = 'train' if split != 'test' else 'test'
            dataset = datasets.load_dataset('amazon_polarity')[split_]

            if split in ['train', 'test']:
                start_idx = 0
            elif split == 'dev':
                start_idx = dataset_info[args.dataset][split_][1]

            for idx in tqdm(range(start_idx, start_idx+num_examples), desc=f'Building {args.split} dataset'):
                text = f'{dataset[idx]["title"]} {tokenizer.sep_token} {dataset[idx]["content"]}'
                tokens = tokenizer(
                    text,
                    padding='max_length',
                    max_length=max_length,
                    truncation=True
                )

                input_ids_ = tokenizer(text, add_special_tokens=False)['input_ids']
                num_tokens = len(input_ids_) + num_special_tokens
                if num_tokens > actual_max_length:
                    actual_max_length = num_tokens

                rand_rationale = list(np.random.randn(max_length))

                dataset_dict['item_idx'].append(idx-start_idx)
                dataset_dict['input_ids'].append(tokens['input_ids'])
                dataset_dict['attention_mask'].append(tokens['attention_mask'])
                dataset_dict['rationale'].append(None)
                dataset_dict['rand_rationale'].append(rand_rationale)
                dataset_dict['has_rationale'].append(0)
                dataset_dict['label'].append(dataset[idx]['label'])

        elif args.dataset == 'olid':
            dataset = datasets.load_dataset('tweet_eval', 'offensive')[split]
            for idx in tqdm(range(num_examples), desc=f'Building {args.split} dataset'):
                text = dataset[idx]['text']

                input_ids_ = tokenizer(text, add_special_tokens=False)['input_ids']
                num_tokens = len(input_ids_) + num_special_tokens
                if num_tokens > actual_max_length:
                    actual_max_length = num_tokens

                rand_rationale = list(np.random.randn(max_length))

                tokens = tokenizer(text, padding='max_length', max_length=max_length, truncation=True)
                dataset_dict['item_idx'].append(idx)
                dataset_dict['input_ids'].append(tokens['input_ids'])
                dataset_dict['attention_mask'].append(tokens['attention_mask'])
                dataset_dict['rationale'].append(None)
                dataset_dict['rand_rationale'].append(rand_rationale)
                dataset_dict['has_rationale'].append(0)
                dataset_dict['label'].append(dataset[idx]['label'])

        elif args.dataset == 'irony':
            dataset = datasets.load_dataset('tweet_eval', 'irony')[split]
            for idx in tqdm(range(num_examples), desc=f'Building {args.split} dataset'):
                text = dataset[idx]['text']

                input_ids_ = tokenizer(text, add_special_tokens=False)['input_ids']
                num_tokens = len(input_ids_) + num_special_tokens
                if num_tokens > actual_max_length:
                    actual_max_length = num_tokens

                rand_rationale = list(np.random.randn(max_length))

                tokens = tokenizer(text, padding='max_length', max_length=max_length, truncation=True)
                dataset_dict['item_idx'].append(idx)
                dataset_dict['input_ids'].append(tokens['input_ids'])
                dataset_dict['attention_mask'].append(tokens['attention_mask'])
                dataset_dict['rationale'].append(None)
                dataset_dict['rand_rationale'].append(rand_rationale)
                dataset_dict['has_rationale'].append(0)
                dataset_dict['label'].append(dataset[idx]['label'])

        else:
            raise NotImplementedError

        print(f'Actual max length: {actual_max_length}')

    else:
        dataset_dict = load_dataset(data_path)

    if args.num_samples is not None:
        assert all([os.path.exists(os.path.join(data_path, f'{x}.pkl')) for x in data_keys])
        dataset_dict = sample_dataset(data_path, dataset_dict, args.split, args.num_samples, args.seed)

    if args.pct_train_rationales is not None:
        assert args.split == 'train'
        sample_rationale_indices(data_path, dataset_dict, num_examples, args.pct_train_rationales, args.seed)
        sys.exit()

    save_dataset(data_path, dataset_dict, args.split, args.num_samples, args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Root directory for datasets')
    parser.add_argument('--dataset', type=str,
                        choices=['cose', 'esnli', 'movies', 'multirc', 'sst', 'amazon', 'yelp', 'stf', 'olid', 'irony'])
    parser.add_argument('--arch', type=str, default='google/bigbird-roberta-base', choices=['google/bigbird-roberta-base', 'bert-base-uncased'])
    parser.add_argument('--split', type=str, help='Dataset split', choices=['train', 'dev', 'test'])
    parser.add_argument('--num_samples', type=int, default=None, help='Number of examples to sample. None means all available examples are used.')
    parser.add_argument('--pct_train_rationales', type=float, default=None, help='Percentage of train examples to provide gold rationales for. None means all available train examples are used.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    main(args)
