# -*- coding: utf-8 -*-
# __author__ = 'K_HOLMES_'
# __time__   = '2019/7/30 14:01'

import pandas as pd
import numpy as np
import os, json

Base_Path = '../dataset/pretrain_data'


def statistics_length():
    for data_tag in ['6w', '40w']:
        file_name = "math_process_{}_id.csv".format(data_tag)
        file_path = os.path.join(Base_Path, file_name)
        math_data_pd = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        id_list = math_data_pd.iloc[:, 0].values
        q_list = math_data_pd.iloc[:, 1].values
        length_list = []
        for q in q_list:
            q = q.replace('\\',"").replace('frac', '').replace('^', '').replace('sqrt', '')
            length_list.append(len(q))
        length_list = np.array(length_list)
        new_pd = {'id':id_list, 'length':length_list}
        pd.DataFrame(new_pd).to_csv(os.path.join(Base_Path, 'statistics', 'math_{}_length.csv'.format(data_tag)), index=None, encoding='utf-8')
    return


def statistic_entity():
    for data_tag in ['6w']:
        file_name = "id_match_entity_{}.json".format(data_tag)
        file_path = os.path.join(Base_Path, file_name)
        math_data = json.load(open(file_path, 'r'))
        id_list, entity_num_list = [], []
        for id, entity_list in math_data.items():
            id_list.append(id)
            entity_num_list.append(len(entity_list))
        new_pd = {'id':id_list, 'num_entity':entity_num_list}
        pd.DataFrame(new_pd).to_csv(os.path.join(Base_Path, 'statistics', 'math_{}_entity_num.csv'.format(data_tag)), index=None, encoding='utf-8')
    return


def divide_section():
    for data_tag in ['6w']:
        math_data = pd.read_csv(os.path.join(Base_Path, 'statistics', 'math_{}_entity_num.csv'.format(data_tag)),
                                encoding='utf_8')
        id_array, length_array = math_data.iloc[:, 0].values, math_data.iloc[:, 1].values
        data_combined = list(zip(id_array, length_array))
        data_sorted = sorted(data_combined, key=lambda x: x[1])
        max_num = data_sorted[-1][1]
        total_size = len(data_sorted)
        print("Size", total_size)
        print("max length:{}, Min:{}, Mean:{}, Std:{}".format(max_num,
                                                              data_sorted[0][1],
                                                              np.mean(length_array),
                                                              np.std(length_array)))
        section_num = 6
        section_size = max_num // section_num
        section_store_id = {i: [] for i in range(section_num)}
        start_id = 0

        for sec_num in range(section_num):
            current_bound = (sec_num + 1) * section_size if sec_num + 1 != section_num else max_num
            print("Current Bound", current_bound)

            while data_sorted[start_id][1] <= current_bound:
                t_length = data_sorted[start_id][1]
                tid = data_sorted[start_id][0]
                section_store_id[sec_num].append(tid)  # store id
                start_id += 1
                if (start_id + 1) >= total_size:
                    break

        for i in range(section_num):
            print(len(section_store_id[i]))


if __name__ == '__main__':
    # statistics_length()
    statistic_entity()
    divide_section()