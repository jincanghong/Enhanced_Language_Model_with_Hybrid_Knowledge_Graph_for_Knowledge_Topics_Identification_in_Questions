# -*- coding: utf-8 -*-
# __author__ = 'K_HOLMES_'
# __time__   = '2019/8/5 15:19'

import os, json, argparse
import pandas as pd, numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from math_bert.novelty_modeling import BertModel as MathBert
from sklearn import manifold
import matplotlib.pyplot as plt


def convert_token2id(tokenizer, token):
    token = tokenizer.tokenize(token)
    token_id_list = tokenizer.convert_tokens_to_ids(token)
    return token_id_list


def load_embedding_layer():
    if 'kg' in args.bert_model:
        pretrained_model = BertModel.from_pretrained(args.bert_model)
    else:
        pretrained_model = MathBert.from_pretrained(args.bert_model)
    layer_name, embedding_layer = list(pretrained_model.named_parameters())[0]
    print(layer_name, embedding_layer.shape)
    # token_vector = embedding_layer[token_id_list]
    return embedding_layer.detach().numpy()


def visualize_word_embedding():
    file_list = os.listdir(args.word_file_list)
    embedding_layer = load_embedding_layer()  # get the embedding layer
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=False)
    total_words_embedding = []
    label2word_index = {ind: [] for ind in range(len(file_list))}  # label: word_index_list
    last_token_list_size = 0
    for find, file in enumerate(file_list):
        file_path = os.path.join(args.word_file_list, file)
        tokens = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
        # get the embedding result of each word
        for token in tokens:
            token_id_list = convert_token2id(tokenizer, token)
            embedding_res = embedding_layer[token_id_list]
            embedding_res = np.sum(embedding_res, axis=0)
            total_words_embedding.append(embedding_res)
        label2word_index[find] += [ind for ind in range(last_token_list_size, len(total_words_embedding)) ]
        last_token_list_size = len(total_words_embedding)
    visualize(total_words_embedding, label2word_index)
    return 0


def visualize_question_embedding():
    file_list = os.listdir(args.question_file_list)
    label2question_index = {} # label: word_index_list
    total_question_embedding = None
    last_token_list_size = 0
    file_class = 'test'
    cnt = 0
    for find, file in enumerate(file_list):
        if file_class == 'test':
            if file_class not in file: continue
        else:
            if 'test' in file: continue
        label2question_index[cnt] = []
        tmp_data = np.load(os.path.join(args.question_file_list, file))
        total_question_embedding = np.concatenate([total_question_embedding, tmp_data], axis=0) \
            if total_question_embedding is not None else tmp_data
        label2question_index[cnt] += [ind for ind in range(last_token_list_size, len(total_question_embedding)) ]
        cnt += 1
        last_token_list_size = len(total_question_embedding)

    print(total_question_embedding.shape)
    model_format = args.question_file_list.split('/')[-1]
    visualize(total_question_embedding, label2question_index, file='question_{}_{}'.format(file_class, model_format))


def visualize(data_x, label2word_index, file='words'):
    n_components = 2
    tsne = manifold.TSNE(n_components=n_components)
    Y = tsne.fit_transform(data_x)
    color_list = ['green', 'yellow', 'blue', 'black', 'gray', 'brown']
    # plt.figure(figsize=(8.5, 4))
    # plt.subplot(1, 1, 1)
    for label, word_index in label2word_index.items():
        plt.scatter(Y[word_index, 0], Y[word_index, 1])
    print(file)
    plt.savefig(os.path.join(Base_Root, '{}.jpg'.format(file)))
    plt.show()


def get_clustering_data_according_laebls():
    Base_Path = '/home/cdk/python_project/bert_multi_tag_cls/dataset/my_raw/'
    file_name = 'test.csv'
    file_path = os.path.join(Base_Path, file_name)
    math_data_pd = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    math_columns = list(math_data_pd.columns)
    columns2int = {col: ind for ind, col in enumerate(math_columns)}

    choosen_label = [line.strip().split(',')[0] for line in
                     open(os.path.join(Base_Path, 'label_data', 'labels4clustering.txt'),
                          encoding='utf-8').readlines()]
    choosen_math_data_index = []
    choosen_label_index = {}
    for label in choosen_label:
        tmp_data = math_data_pd.iloc[:, columns2int[label]].values
        tmp_index = list(np.where(tmp_data == 1)[0].astype(float))
        choosen_math_data_index += tmp_index
        choosen_label_index[label] = tmp_index
        print(len(tmp_index))
        pd.DataFrame(math_data_pd.iloc[tmp_index, :].values).to_csv(
            os.path.join(Base_Path, 'label_data','test_data' ,'label_{}_test.csv'.format(label)),
            encoding='utf_8_sig', index=None, header=math_columns, sep='\t')


if __name__ == '__main__':
    Base_Root = '/home/cdk/python_project/bert_multi_tag_cls/dataset/all_data_tags/labels4clustering'
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_tokenizer", default='/home/cdk/python_project/bert_multi_tag_cls/model/pytorch_pretrain_chinese',
                        type=str, required=False)
    parser.add_argument("--bert_model", default='/home/cdk/python_project/bert_multi_tag_cls/model/math_used_pretrain_model/kg_nsp_novelty_6w_pretrain_model_0724',
                        type=str, required=False)
    parser.add_argument("--word_file_list", default='/home/cdk/python_project/bert_multi_tag_cls/dataset/all_data_tags/labels4clustering/keywords',
                        type=str, required=False)
    parser.add_argument("--question_file_list", default='/home/cdk/python_project/bert_multi_tag_cls/dataset/my_raw/label_data_embdding/ours',
                        type=str, required=False)

    args = parser.parse_args()
    # visualize_word_embedding()
    visualize_question_embedding()
    # get_clustering_data_according_laebls()