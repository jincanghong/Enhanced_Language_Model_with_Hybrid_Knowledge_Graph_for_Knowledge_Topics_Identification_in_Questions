# -*- coding: utf-8 -*-
# __author__ = 'K_HOLMES_'
# __time__   = '2019/6/28 9:22'

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
import os

root_path = '/home/cdk/python_project/bert_multi_tag_cls/dataset/all_data_tags'
jieba.load_userdict(os.path.join(root_path, 'math.txt'))


def question_classify():
    # data_file = ['test.csv', 'train.csv']
    file_list = os.listdir(os.path.join(root_path, 'labels4clustering'))
    for file in file_list:
        print(file)
        file_label = file.split('_')[1]
        data_path = os.path.join(root_path, 'labels4clustering', file)
        data = pd.read_csv(data_path, sep='\t', encoding='utf-8')
        header = data.columns
        tags_colums = np.array(header[2:])
        tag2question_dict = {key:[] for key in tags_colums}
        for ind in range(len(data)):
            question = data.iloc[ind, 1]
            try:
                question = ' '.join(jieba.cut(question, cut_all=False))
            except Exception as e:
                # print(line[0], line[1], np.where(line[2:] == 1))
                print(data.iloc[ind, 0])
                continue
            tags = data.iloc[ind, 2:]
            is_1_index = np.where(tags == 1)
            tags_name = list(tags_colums[is_1_index])
            for each_tag in tags_name:
                if each_tag not in file_label:continue
                tag2question_dict[each_tag].append(question)
        for key, value in tag2question_dict.items():
            if len(value) == 0:continue
            with open(os.path.join(root_path, 'classification_documents/{}.txt'.format(key)), 'a+', encoding='utf-8') as f:
                for question in value:
                    f.write('{}\n'.format(question))


def cal_tf_idf_scikit():
    documents_path = os.path.join(root_path, 'classification_documents')
    file_list = sorted(os.listdir(documents_path))
    for file in file_list:
        corpus = []
        file_path = os.path.join(documents_path, file)
        corpus.append(open(file_path).read())
        vectorizer = CountVectorizer(stop_words=['如图','学校', '学生', '0'])
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        word = vectorizer.get_feature_names()  # 所有文本的关键字

        weight = tfidf.toarray()[0]  # 对应的tfidf矩阵
        weight_index = np.argsort(weight)[::-1][:100]
        print(weight.shape, weight[weight_index])

        with open(os.path.join(root_path, 'tf_idf', 'word_extraction_{}.txt'.format(file.replace('.txt', ''))), 'w')as f:
            tmp_word = np.array(word)[weight_index]
            print(len(tmp_word), tmp_word)
            for each in tmp_word:
                f.write("{}\n".format(each))
        # for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重
        #     np.save(os.path.join(root_path, 'tf_idf', file_list[i].replace('.txt', '')), weight[i])
            # pd.DataFrame(weight[i]).to_csv(os.path.join(root_path, 'tf_idf', file_list[i].replace('.txt', '.csv')),
            #                                index=None, header=word)


if __name__ == '__main__':
    # question_classify()
    cal_tf_idf_scikit()