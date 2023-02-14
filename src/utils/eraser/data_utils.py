import logging
from typing import Tuple, List, Dict, Any
from collections import namedtuple, defaultdict
from itertools import chain
from tokenizers import TextInputSequence

from src.utils.eraser.utils import Annotation, Evidence, annotations_from_jsonl, load_documents

SentenceEvidence = namedtuple('SentenceEvidence', 'kls ann_id query docid index sentence')

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)


def bert_tokenize_doc(doc: List[List[str]], tokenizer, special_token_map) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
    """ Tokenizes a document and returns [start, end) spans to map the wordpieces back to their source words"""
    sents = []
    sent_token_spans = []
    for sent in doc:
        tokens = []
        spans = []
        start = 0
        for w in sent:
            if w in special_token_map:
                tokens.append(w)
            else:
                tokens.extend(tokenizer.tokenize(w))
            end = len(tokens)
            spans.append((start, end))
            start = end
        sents.append(tokens)
        sent_token_spans.append(spans)
    return sents, sent_token_spans


def bert_intern_doc(doc: List[List[str]], tokenizer, special_token_map) -> List[List[int]]:
    # return [list(chain.from_iterable(special_token_map.get(w, tokenizer.encode(w)) for w in s)) for s in doc]
    return [[special_token_map.get(w, tokenizer.convert_tokens_to_ids(w)) for w in s] for s in doc]


def bert_intern_annotation(annotations: List[Annotation], tokenizer):
    ret = []
    for ann in annotations:
        ev_groups = []
        for ev_group in ann.evidences:
            evs = []
            for ev in ev_group:
                text = list(chain.from_iterable(tokenizer.tokenize(w) for w in ev.text.split()))
                if len(text) == 0:
                    continue
                # text = tokenizer.encode(text, add_special_tokens=False)
                text = tokenizer.convert_tokens_to_ids(text)
                evs.append(Evidence(text=tuple(text),
                                    docid=ev.docid,
                                    start_token=ev.start_token,
                                    end_token=ev.end_token,
                                    start_sentence=ev.start_sentence,
                                    end_sentence=ev.end_sentence))
            ev_groups.append(tuple(evs))
        query = list(chain.from_iterable(tokenizer.tokenize(w) for w in ann.query.split()))
        if len(query) > 0:
            # query = tokenizer.encode(query, add_special_tokens=False)
            query = tokenizer.convert_tokens_to_ids(query)
        else:
            query = []
        ret.append(Annotation(annotation_id=ann.annotation_id,
                              query=tuple(query),
                              evidences=frozenset(ev_groups),
                              classification=ann.classification,
                              query_type=ann.query_type))
    return ret


def annotations_to_evidence_identification(annotations: List[Annotation],
                                           documents: Dict[str, List[List[Any]]]
                                           ) -> Dict[str, Dict[str, List[SentenceEvidence]]]:
    """Converts Corpus-Level annotations to Sentence Level relevance judgments.

    As this module is about a pipelined approach for evidence identification,
    inputs to both an evidence identifier and evidence classifier need to be to
    be on a sentence level, this module converts data to be that form.

    The return type is of the form
        annotation id -> docid -> [sentence level annotations]
    """
    ret = defaultdict(dict)  # annotation id -> docid -> sentences
    for ann in annotations:
        ann_id = ann.annotation_id
        for ev_group in ann.evidences:
            for ev in ev_group:
                if len(ev.text) == 0:
                    continue
                if ev.docid not in ret[ann_id]:
                    ret[ann.annotation_id][ev.docid] = []
                    # populate the document with "not evidence"; to be filled in later
                    for index, sent in enumerate(documents[ev.docid]):
                        ret[ann.annotation_id][ev.docid].append(SentenceEvidence(
                            kls=0,
                            query=ann.query,
                            ann_id=ann.annotation_id,
                            docid=ev.docid,
                            index=index,
                            sentence=sent))
                # define the evidence sections of the document
                for s in range(ev.start_sentence, ev.end_sentence):
                    ret[ann.annotation_id][ev.docid][s] = SentenceEvidence(
                        kls=1,
                        ann_id=ann.annotation_id,
                        query=ann.query,
                        docid=ev.docid,
                        index=ret[ann.annotation_id][ev.docid][s].index,
                        sentence=ret[ann.annotation_id][ev.docid][s].sentence)
    return ret


def annotations_to_evidence_token_identification(annotations: List[Annotation],
                                                 source_documents: Dict[str, List[List[str]]],
                                                 interned_documents: Dict[str, List[List[int]]],
                                                 token_mapping: Dict[str, List[List[Tuple[int, int]]]]
                                                 ) -> Dict[str, Dict[str, List[SentenceEvidence]]]:
    # TODO document
    # TODO should we simplify to use only source text?
    ret = defaultdict(lambda: defaultdict(list)) # annotation id -> docid -> sentences
    positive_tokens = 0
    negative_tokens = 0
    for ann in annotations:
        annid = ann.annotation_id
        docids = set(ev.docid for ev in chain.from_iterable(ann.evidences))
        sentence_offsets = defaultdict(list) # docid -> [(start, end)]
        classes = defaultdict(list) # docid -> [token is yea or nay]
        for docid in docids:
            start = 0
            assert len(source_documents[docid]) == len(interned_documents[docid])
            for whole_token_sent, wordpiece_sent in zip(source_documents[docid], interned_documents[docid]):
                classes[docid].extend([0 for _ in wordpiece_sent])
                end = start + len(wordpiece_sent)
                sentence_offsets[docid].append((start, end))
                start = end
        for ev in chain.from_iterable(ann.evidences):
            if len(ev.text) == 0:
                continue
            flat_token_map = list(chain.from_iterable(token_mapping[ev.docid]))
            if ev.start_token != -1:
                #start, end = token_mapping[ev.docid][ev.start_token][0], token_mapping[ev.docid][ev.end_token][1]
                start, end = flat_token_map[ev.start_token][0], flat_token_map[ev.end_token - 1][1]
            else:
                start = flat_token_map[sentence_offsets[ev.start_sentence][0]][0]
                end = flat_token_map[sentence_offsets[ev.end_sentence - 1][1]][1]
            for i in range(start, end):
                classes[ev.docid][i] = 1
        for docid, offsets in sentence_offsets.items():
            token_assignments = classes[docid]
            positive_tokens += sum(token_assignments)
            negative_tokens += len(token_assignments) - sum(token_assignments)
            for s, (start, end) in enumerate(offsets):
                sent = interned_documents[docid][s]
                ret[annid][docid].append(SentenceEvidence(kls=tuple(token_assignments[start:end]),
                                                          query=ann.query,
                                                          ann_id=ann.annotation_id,
                                                          docid=docid,
                                                          index=s,
                                                          sentence=sent))
    logging.info(f"Have {positive_tokens} positive wordpiece tokens, {negative_tokens} negative wordpiece tokens")
    return ret