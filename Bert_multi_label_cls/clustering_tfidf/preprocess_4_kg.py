# -*- coding: utf-8 -*-
# __author__ = 'K_HOLMES_'
# __time__   = '2019/7/10 12:21'

import pandas as pd
import os

root_path = './'
file_name = 'Entity_relation_0716.xlsx'

data = pd.read_excel(os.path.join(root_path, file_name), header=None)

relation_list = sorted(set(list(data[1])))
entity_list = sorted(set(list(data[0])) | set(list(data[2])))
print("Total relation:{}, entity:{}".format(len(relation_list), len(entity_list)))
# print("Relation size", len(relation_list))
# print("Entity size", len(entity_list))
relation2id_dict = {}
entity2id_dict = {}
with open(os.path.join(root_path, 'math_kg', 'relation2id.txt'), 'w', encoding='utf-8') as f:
    f.write("{}\n".format(len(relation_list)))
    for ind, val in enumerate(relation_list):
        f.write("{}\t{}\n".format(val, ind))
        relation2id_dict[val] = ind

with open(os.path.join(root_path, 'math_kg', 'd.txt'), 'w', encoding='utf-8') as f:
    f.write("{}\n".format(len(entity_list)))
    for ind, val in enumerate(entity_list):
        f.write("{}\t{}\n".format(val, ind))
        entity2id_dict[val] = ind
train_data = []
total_size = len(data)

for row in range(total_size):
    line = data.iloc[row]
    train_data.append([entity2id_dict[line[0]], entity2id_dict[line[2]], relation2id_dict[line[1]]])

with open(os.path.join(root_path, 'math_kg', 'train2id.txt'), 'w', encoding='utf-8') as f:
    f.write("{}\n".format(len(train_data)))
    for ind, val in enumerate(train_data):
        f.write("{}\t{}\t{}\n".format(val[0], val[1], val[2]))

print("Total train size:{}".format(len(train_data)))