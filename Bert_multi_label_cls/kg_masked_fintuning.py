# -*- coding: utf-8 -*-
# __author__ = 'K_HOLMES_'
# __time__   = '2019/7/20 12:42'


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import json
from tqdm import tqdm, trange
from itertools import chain

import numpy as np
import torch
import networkx as nx

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from math_bert.novelty_modeling import BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import Dataset

import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True,
                 id_match_entity_path=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line
        self.id_match_entity = json.load(open(os.path.join(id_match_entity_path), 'r', encoding=encoding))
        entity2id = {}
        with open("clustering_tfidf/math_kg/entity2id.txt", encoding='utf-8') as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                entity2id[qid] = int(eid)
        self.entity2id = entity2id
        self.t_id2int = {}  # 题目id转int
        self.int2t_id = {}  # int转题目id
        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for ind, line in enumerate(tqdm(f, desc="Loading Dataset", total=corpus_lines)):
                    line = line.strip().split('\t')
                    if len(line) == 2:
                        t_id, content = line[0], line[1]
                        self.t_id2int[t_id] = ind
                        self.int2t_id[ind] = t_id
                    else:
                        t_id, content = "", ""
                    if content == "":
                        self.all_docs.append(doc)
                        doc = []
                        # remove last added sample because there won't be a subsequent line anymore in the doc
                        self.sample_to_doc.pop()
                    else:
                        # store as one sample, mark this line belongs to which doc
                        sample = {"doc_id": len(self.all_docs),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append((t_id, content))
                        self.corpus_lines = self.corpus_lines + 1
            # if last row in file is not empty
            if len(self.all_docs) == 0 or self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_docs)
            print("Length", self.num_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        return self.corpus_lines - self.num_docs

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        t1 = self.random_sent(item)
        t_id, content = t1[0], t1[1]
        # tokenize
        tokens_a = self.tokenizer.tokenize(content)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=None, tokens_a_id=t_id)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.id_match_entity,
                                                   self.entity2id, self.t_id2int)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.input_ent),
                       torch.tensor(cur_features.ent_mask),
                       torch.tensor(cur_features.t_id),
                       )

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1 = self.get_corpus_line(index)
        assert len(t1) > 0
        return t1

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "" :
                    t1 = self.file.__next__().strip()
            else:
                t1 = self.line_buffer
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t1 == "":
                    t1 = self.file.__next__().strip()
                    self.current_doc = self.current_doc+1
            self.line_buffer = t1

        assert t1 != ""
        return t1


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, lm_labels=None, tokens_a_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_a_id = tokens_a_id
        self.tokens_b = tokens_b
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, lm_label_ids, input_ent, ent_mask, t_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.input_ent = input_ent
        self.ent_mask = ent_mask
        self.t_id = t_id  # question id


def random_word(tokens, tokenizer, id_match_entity, t_id):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    # 如果有entity  则mask的是他的entity 部分
    prob_entity_mask = random.random()
    entity_list = id_match_entity[t_id] if t_id in id_match_entity else []  # id_match_entity only record non empty id
    skip_char = [chr(i) for i in range(97,123)] + [chr(i) for i in range(65, 91)] + ['\\']
    if prob_entity_mask > 0.5 or len(entity_list) == 0: # if this question has no entity also use this
        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                # Todo 不去mask entity部分的内容 英文特殊考虑
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(tokenizer.vocab["[UNK]"])
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
    else:
        len_tokens = len(tokens)
        tmp_label = [-1] * len_tokens # 1 mask; -1 not mask
        # for each entity mask its correpsonding entity or mask partial
        prob_list = [random.random() for _ in entity_list]
        skip_ind = -1
        for i in range(len_tokens):
            if i < skip_ind:
                continue  # keep re_calculate index
            for e_ind, entity in enumerate(entity_list):
                left_ind = min(i+len(entity), len_tokens)
                combine_str = ''.join(tokens[i:left_ind])
                # print(combine_str, entity)
                if tokens[i] in entity and combine_str == entity:  # equal
                    if prob_list[e_ind] < 0.5:  # mask all of them
                        for j in range(i, left_ind):
                            tokens[j] = "[MASK]"
                            tmp_label[j] = 1
                    else:
                        tmp_prob = prob_list[e_ind] / 1
                        for j in range(i, left_ind):
                            if tmp_prob < 0.8:  # mask all of them
                                if random.random() < 0.5:
                                    tokens[j] = "[MASK]"
                                    tmp_label[j] = 1
                            elif tmp_prob < 0.9:
                                if random.random() < 0.5:  # mask all of them
                                    tokens[j] = random.choice(list(tokenizer.vocab.items()))[0]
                                    tmp_label[j] = 1
                    skip_ind = left_ind
                    break
        assert len(tmp_label) == len(tokens)
        for tmp, token in zip(tmp_label, tokens):
            if tmp == -1:
                output_label.append(-1)
            else:
                try:
                    output_label.append(tokenizer.vocab[token])
                except KeyError:
                    output_label.append(tokenizer.vocab["[UNK]"])
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer, id_match_entity, entity2id, t_id2int):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :param id_match_entity:
    :param entity2id:
    :param t_id2int: a dict which convert a id of question to int number
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)

    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    t_id = example.tokens_a_id
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    # Entity size

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with '-2'-
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
    ENTITY_LENGTH = 50
    input_ent = []
    ent_mask = []
    if t_id in id_match_entity:
        for ent in id_match_entity[t_id]:
            if ent != "UNK" and ent in entity2id:
                input_ent.append(entity2id[ent])
                ent_mask.append(1)


    tokens_a, t1_label = random_word(tokens_a, tokenizer, id_match_entity, t_id)
    if tokens_b:
        tokens_b, t2_label = random_word(tokens_b, tokenizer, id_match_entity, t_id)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    # for the single sentence, the segment_ids are all 0
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    # assert len(tokens_b) > 0
    # for token in tokens_b:
    #     tokens.append(token)
    #     segment_ids.append(1)
    # tokens.append("[SEP]")
    # segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    while len(input_ent) < ENTITY_LENGTH:
        input_ent.append(-1)
        ent_mask.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    assert len(input_ent) == ENTITY_LENGTH
    assert len(ent_mask) == ENTITY_LENGTH

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))
        logger.info("Input Entity: %s " % (input_ent))
        logger.info("Entity Mask: %s " % (ent_mask))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             input_ent=input_ent,
                             ent_mask=ent_mask,
                             t_id=t_id2int[t_id])
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default='/home/cdk/python_project/bert_multi_tag_cls/dataset/pretrain_data/math_process_6w_id.csv',
                        type=str,
                        required=False,
                        help="The input train corpus.")
    parser.add_argument("--bert_model", default='/home/cdk/python_project/bert_multi_tag_cls/model/pytorch_pretrain_chinese', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    '''
    novelty部分只增加了maskLM操作 ， kg_novelty 增加了MaskedML与kg在model的引入
    '''
    parser.add_argument("--output_dir",
                        default='/home/cdk/python_project/bert_multi_tag_cls/model/math_all_pretrain_model/kg_novelty_math_6w_save_pretrain_model_0721',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--id_match_entity_path",
                        default='/home/cdk/python_project/bert_multi_tag_cls/dataset/pretrain_data/id_match_entity_6w.json',
                        type=str,
                        required=False,
                        help="The file of id match entity")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=544,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        default=True,
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    print(tokenizer)
    #train_examples = None
    num_train_optimization_steps = None
    print("Loading Train Dataset", args.train_file)
    train_dataset = BERTDataset(args.train_file, tokenizer, seq_len=args.max_seq_length,
                                corpus_lines=None, on_memory=args.on_memory, id_match_entity_path=args.id_match_entity_path)
    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    vecs = []
    vecs.append([0] * 100)
    with open("clustering_tfidf/math_kg/entity2vec.vec", 'r') as fin:
        for ind, line in enumerate(fin):
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed = torch.FloatTensor(vecs)
    embed = torch.nn.Embedding.from_pretrained(embed).to(device)

    G = nx.Graph()
    entity2id = train_dataset.entity2id # '三角形' -> 2
    entity2id['∥'] = 34
    entity2id['｜'] = 449
    int2t_id = train_dataset.int2t_id # 1 - > 'q2324'
    id_match_entity = train_dataset.id_match_entity # 'q2324' -> ['三角形', '']
    with open("clustering_tfidf/math_kg/train2id.txt", 'r') as fin:
        for ind, line in enumerate(fin):
            if ind == 0: continue
            eid = [int(each) for each in line.strip().split('\t')]
            G.add_edge(eid[0], eid[1])


    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on file.__next__
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
    sample4calculation = 60  # 组数
    upper_sample_bound = args.train_batch_size // n_gpu  # 17
    for ind_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(train_dataloader, desc="Iteration") as tqdm_batch:
            for step, batch in enumerate(tqdm_batch): # batch的问题
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, input_ent, ent_mask, t_id_array = batch
                input_ent = embed(input_ent+1)  # -1 -> 0
                correlation_label = []
                batch_random_tuple4cls = []
                if upper_sample_bound * 4 <= input_ids.shape[0]:
                    batch_random_tuple4cls = torch.unique(torch.randint(0, upper_sample_bound, (sample4calculation*50, 2)), dim=0)
                    while batch_random_tuple4cls.shape[0] < sample4calculation:
                        residual_size = sample4calculation - batch_random_tuple4cls.shape[0]
                        batch_random_tuple4cls = torch.cat([batch_random_tuple4cls,
                                                            torch.unique(torch.randint(0, upper_sample_bound,
                                                                                       (residual_size, 2)), dim=0)], dim=0)
                    batch_random_tuple4cls = batch_random_tuple4cls[:sample4calculation]
                    for tmp in batch_random_tuple4cls:
                        # according to random batch sample index to find which t_id(str)
                        t_id_0, t_id_1 = int2t_id[t_id_array[tmp[0]].item()], int2t_id[t_id_array[tmp[1]].item()]
                        # TODO consider id does not have entity
                        entity_list_0, entity_list_1 = id_match_entity[t_id_0] if t_id_0 in id_match_entity else [], \
                                                       id_match_entity[t_id_1] if t_id_1 in id_match_entity else []
                        entity_list_0 = [list(G[entity2id[entity_0]]) for entity_0 in entity_list_0]
                        entity_list_1 = [list(G[entity2id[entity_1]]) for entity_1 in entity_list_1]
                        entity_list_0 = set(chain(*entity_list_0))
                        entity_list_1 = set(chain(*entity_list_1))
                        union_size = len((entity_list_0 | entity_list_1))
                        jacarrd_score = round(len((entity_list_0 & entity_list_1)) / union_size, 4) if union_size != 0 else 0
                        correlation_label.append(jacarrd_score * 10)

                correlation_label = torch.Tensor(correlation_label).to(device)
                batch_random_tuple4cls = torch.LongTensor(batch_random_tuple4cls).to(device)
                loss = model(input_ids, input_ent, ent_mask, segment_ids, input_mask, lm_label_ids,
                             batch_random_tuple4cls, correlation_label)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                tqdm_batch.set_description("Iteration loss:{}".format(round(loss.item(),4)))
        # Save a trained model
        logger.info("** ** * Saving fine - tuned model, epoch_loss:{} ** ** * ".format(round(tr_loss, 4)))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model_{}_{}.bin".format(ind_epoch, round(tr_loss, 3)))
        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

if __name__ == "__main__":
    main()
