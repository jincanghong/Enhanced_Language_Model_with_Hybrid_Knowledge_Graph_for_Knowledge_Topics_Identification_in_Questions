#encoding:utf-8
import random
import operator
import pandas as pd
from tqdm import tqdm
from collections import Counter
from ..utils.utils import text_write, text_read
from ..utils.utils import pkl_write


class DataTransformer(object):
    def __init__(self,
                 logger,
                 seed,
                 add_unk=True
                 ):
        self.seed = seed
        self.logger = logger
        self.item2idx = {}
        self.idx2item = []
        self.label_size = 23
        # 未知的tokens
        if add_unk:
            self.add_item('<unk>')

    def add_item(self,item):
        '''
        对映射字典中新增item
        :param item:
        :return:
        '''
        item = item.encode('UTF-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1

    def get_idx_for_item(self,item):
        '''
        获取指定item的id，如果不存在，则返回0，即unk
        :param item:
        :return:
        '''
        item = item.encode('UTF-8')
        if item in self.item2idx:
            return self.item2idx[item]
        else:
            return 0

    def get_item_for_index(self, idx):
        '''
        给定id，返回对应的tokens
        :param idx:
        :return:
        '''
        return self.idx2item[idx].decode('UTF-8')

    def get_items(self):
        '''
        获取所有的items
        :return:
        '''
        items = []
        for item in self.idx2item:
            items.append(item.decode('UTF-8'))

    def split_sent(self,line):
        """
        句子处理成单词
        :param line: 原始行
        :return: 单词， 标签
        """
        res = line.strip('\n').split()
        return res

    def train_val_split(self,X, y, question_ids, valid_size,
                        shuffle=True,
                        save = True,
                        train_path = None,
                        valid_path = None):
        '''
        # 将原始数据集分割成train和valid
        :return:
        '''
        self.logger.info('train val split')
        data = []
        for data_x, data_y, q_id in tqdm(zip(X, y, question_ids), desc='Merge'):
            data.append((q_id, data_x, data_y))
        del X, y, question_ids
        N = len(data)
        test_size = int(N * valid_size)
        if shuffle:
            random.seed(self.seed)
            random.shuffle(data)
        valid = data[:test_size]
        train = data[test_size:]
        # 混洗train数据集
        if shuffle:
            random.seed(self.seed)
            random.shuffle(train)
        print(len(train), len(train[0]))
        if save:
            text_write(filename=train_path, data=train)
            text_write(filename=valid_path, data=valid)
        return train, valid

    def build_vocab(self,data,min_freq,max_features,save,vocab_path):
        '''
        建立语料库
        :param data:
        :param min_freq:
        :param max_features:
        :param save:
        :param vocab_path:
        :return:
        '''
        count = Counter()
        self.logger.info('Building word vocab')
        for i,line in enumerate(data):
            words = self.split_sent(line)
            count.update(words)
        count = {k: v for k, v in count.items()}
        count = sorted(count.items(), key=operator.itemgetter(1))
        # 词典
        all_words = [w[0] for w in count if w[1] >= min_freq]
        if max_features:
            all_words = all_words[:max_features]

        self.logger.info('vocab_size is %d' % len(all_words))
        for word in all_words:
            self.add_item(item = word)
        if save:
            # 写入文件中
            pkl_write(data = self.item2idx,filename = vocab_path)

    def read_data(self,raw_data_path,preprocessor=None, is_train=True):
        '''
        读取原始数据集,这里需要根据具体任务的进行相对应的修改
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        data = pd.read_csv(raw_data_path, encoding='utf-8-sig', sep='\t')
        # targets = data.values[:, 2:]
        # sentences = data.values[:, 1]
        # question_id = data.values[:, 0]
        targets, sentences, question_id = [], [], []
        for i, row in enumerate(tqdm(data.values)):
            target = row[2:] if len(row) >= 2 else [0] * 1
            question_id.append(row[0])
            sentence = str(row[1])
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
        return targets, sentences, question_id

    def read_processed_data(self, filename):
        # 读取处理后的数据
        return text_read(filename)
