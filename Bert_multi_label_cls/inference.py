#encoding:utf-8
import torch
import warnings, os, numpy as np,pandas as pd, json
from torch.utils.data import DataLoader
from pybert.io_data.dataset import CreateDataset
from pybert.io_data.data_transformer import DataTransformer
from pybert.utils.logginger import init_logger
from pybert.utils.utils import seed_everything
from pybert.config.basic_config import configs as config, BASE_DIR
from pybert.model.nn.bert_fine import BertFine
from pybert.test.predicter import Predicter
from pytorch_pretrained_bert.tokenization import BertTokenizer
warnings.filterwarnings("ignore")
is_train_data = False

model_checkpoint = 'epoch_50_0.0112_bert_model.pth'
model_checkpoint_file = '{}'.format(model_checkpoint) if not is_train_data else '{}_{}'.format('train', model_checkpoint)
save_path = os.path.join(config['output']['result'], str(config['output']['checkpoint_dir']).split('/')[-1])
res_path = os.path.join(save_path, model_checkpoint_file[:-4]+'_res')
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(res_path):
    os.mkdir(res_path)

tmp_res_path = res_path


def according_index2question():
    question_path = os.path.join(res_path, 'question')
    if not os.path.exists(question_path):
        os.mkdir(question_path)
    total_data = pd.read_csv(os.path.join(BASE_DIR, 'dataset/all_data_tags/clean_math_0731.csv'), sep='\t', encoding='utf_8_sig')
    total_data = total_data.set_index('question_id')
    top_size_list = [5, 3, 1]  # top k shoot
    for top_size in top_size_list:
        file_name = 'record_wrong_prediciton_res_top{}.csv'.format(top_size)
        data = pd.read_csv(os.path.join(res_path, file_name), sep='\t')
        index_col = data['QuestionID']
        questions_col = total_data.ix[index_col, 'stem_content']
        questions_col = pd.DataFrame(questions_col).reset_index()
        questions_col['TrueLabel'] = data['TrueLabel']
        questions_col['Predict'] = data['Predict']
        questions_col.to_csv(os.path.join(question_path, 'record_wrong_prediciton_with_question{}.csv'.format(top_size)),
                             index=None, sep='\t', encoding='utf_8_sig',)
        # print(index_col[:10])


def compare_true_pre(y_true, y_pred, question_id):
    id2label = {value: key for key, value in config['my_label2id'].items()}
    total_size = len(y_true)
    print("Total size:{}, shape:{}".format(total_size, y_pred.shape))
    top_size_list = [5, 3, 1]  # top k shoot
    acc_res_list = []  # return the result txt
    for top_size in top_size_list:
        get_shoot_at_least_1 = 0  # 至少有一个label在该题中被预测对
        total_true_labels, total_predict_right = 0, 0 # 所有题目的标签次数和，对每道题目中标签预测次数
        cnt_appear_label_dict = {label: [0, 0, 0] for key, label in id2label.items()}  # 计数出现的次数的label 和 预测对的label的数量
        right_list, non_right_list = [], []
        false_predict_question_and_predict_id = []
        for ind, cur_pred in enumerate(y_pred):
            # 获得真实值的1的index
            true_index = np.where(y_true[ind]==1)[0]
            # 获得pred的argmax, 取得与实际值同样的size, -cur_pred是为了从大到小
            pred_index = np.argsort(-cur_pred)[:top_size]
            # if ind == 0:
            #     print("Test result\n",true_index,'\n',pred_index)
            intersection = set(true_index) & set(pred_index)
            total_true_labels += len(true_index) if len(true_index) <= top_size else top_size  # 每次只能最多加当前topk的数量
            total_predict_right += len(intersection)
            # mark the appearing times of each label, 标记出现的次数部分
            for each_id in true_index:
                cnt_appear_label_dict[id2label[each_id]][0] += 1
            if len(intersection) > 0:
                get_shoot_at_least_1 += 1
                for each_id in intersection:
                    right_list.append((question_id[ind], id2label[each_id]))
                    cnt_appear_label_dict[id2label[each_id]][1] += 1  # 标记预测对的部分
            else:
                for each_id in true_index:
                    non_right_list.append((question_id[ind], id2label[each_id]))
            # 对每一个没有全部预测出来的题目 进行错题统计 同时要忽略top k设定的影响
            if len(intersection) < len(true_index) and len(true_index) <= top_size:
                right_label, false_label = '', ''
                for each_id in true_index:
                    right_label += id2label[each_id] + ';'
                for each_id in pred_index:
                    false_label += id2label[each_id] + ';'
                false_predict_question_and_predict_id.append([question_id[ind], right_label[:-1], false_label[:-1]])
        for key in cnt_appear_label_dict.keys():
            cnt_appear_label_dict[key][2] = round(cnt_appear_label_dict[key][1] / cnt_appear_label_dict[key][0]
                                                  if cnt_appear_label_dict[key][0] else 0, 3)
        # 根据预测准群率从高到底
        cnt_appear_label_dict = sorted(cnt_appear_label_dict.items(), key=lambda item:item[1][2], reverse=True)
        with open(os.path.join(res_path, 'appear_predict_right_rate_top{}.txt'.format(top_size)), 'w') as f:
            f.write("Label_name,Appearance_Times,Predict_Right_Times\n")
            for each in cnt_appear_label_dict:
                key, value = each[0], each[1]
                f.write("{},{},{},{}\n".format(key, value[0], value[1], value[2]))

        pd.DataFrame(right_list).to_csv(os.path.join(res_path, 'predict_true_top{}.csv'.format(top_size)),
                                        index=None,
                                        header=['QuestionId', 'Label'], encoding='utf_8_sig')
        pd.DataFrame(non_right_list).to_csv(os.path.join(res_path, 'predict_false_top{}.csv'.format(top_size)),
                                        index=None,
                                        header=['QuestionId', 'Label'], encoding='utf_8_sig')
        with open(os.path.join(res_path, 'record_wrong_prediciton_res_top{}.csv'.format(top_size)), 'w', encoding='utf_8_sig')as f:
            f.write("QuestionID\tTrueLabel\tPredict\n")
            for line in false_predict_question_and_predict_id:
                f.write("{}\t{}\t{}\n".format(line[0], line[1], line[2]))
        with open(os.path.join(res_path, 'total_result.txt'.format(top_size)), 'a+') as f:
            res_ = "At least one right prediction in each question: {} in Top {}, ACC:{}%".\
                format(get_shoot_at_least_1, top_size, (get_shoot_at_least_1 / total_size) * 100)
            f.write(res_ + '\n')
            print(res_)
            res_ = "In Top{}, the number of total labels in all questions are :{},  right prediction numbers are:{}, ACC:{}%".\
                format(top_size, total_true_labels, total_predict_right, (total_predict_right/ total_true_labels) * 100)
            acc_res_list.append(round(total_predict_right / total_true_labels, 4) * 100)
            f.write(res_ + '\n')
            print(res_)

    return acc_res_list


def cal_p_f_r():
    save_dict = torch.load(os.path.join(res_path, 'predict.pt'))
    y_true, y_pred, question_id = save_dict['y_true'], save_dict['y_pred'], save_dict['question_id']
    compare_true_pre(y_true, y_pred, question_id, )


# 主函数
def main():
    # **************************** 基础信息 ***********************
    logger = init_logger(log_name=config['model']['arch'], log_dir=config['output']['log_dir'])
    logger.info(f"seed is {config['train']['seed']}")
    device = 'cuda:%d' % config['train']['n_gpu'][0] if len(config['train']['n_gpu']) else 'cpu'
    seed_everything(seed=config['train']['seed'],device=device)
    logger.info('starting load data from disk')
    id2label = {value: key for key, value in config['my_label2id'].items()}

    #**************************** 数据生成 ***********************
    DT = DataTransformer(logger=logger, seed=config['train']['seed'])


    # **************************** Preparing Data ***********************

    if config['data']['test_length_dataset']:  # Need to test data set divided by length
        test_file_list = sorted([os.path.join(config['data']['test_length_dataset_path'], each)
                                 for each in os.listdir(config['data']['test_length_dataset_path'])])
    else:
        test_file_list = config['data']['test_file_path']

    # **************************** 模型 ***********************
    logger.info("initializing model")
    model = BertFine.from_pretrained(config['pretrained']['bert']['bert_model_dir'],
                                     cache_dir=config['output']['cache_dir'],
                                     num_classes = len(id2label))
    # **************************** training model ***********************
    logger.info('model predicting....')
    predicter = Predicter(model = model,
                         logger = logger,
                         n_gpu=config['train']['n_gpu'],
                         model_path =os.path.join(config['output']['checkpoint_dir'], model_checkpoint),
                         )
    f_store_res = open(os.path.join(tmp_res_path, 'store_test_set_res_{}.txt'.format('length')), 'w', encoding='utf-8')
    # 读取数据集以及数据划分
    for test_file in test_file_list:
        targets, sentences, question_id = DT.read_data(raw_data_path=test_file,
                                          is_train=False)
        tokenizer = BertTokenizer(vocab_file=config['pretrained']['bert']['vocab_path_ch'],
                                  do_lower_case=config['train']['do_lower_case'])
        # test
        test_dataset   = CreateDataset(data=list(zip(question_id,sentences,targets)),
                                       tokenizer=tokenizer,
                                       max_seq_len=config['train']['max_seq_len'],
                                       seed=config['train']['seed'],
                                       example_type='test',
                                       )
        test_loader = DataLoader(dataset     = test_dataset,
                                 batch_size  = config['predict']['batch_size'],
                                 num_workers = config['train']['num_workers'],
                                 shuffle     = False,
                                 drop_last   = False,
                                 pin_memory  = False)
        global res_path
        test_name = str(test_file).split('/')[-1].replace(".csv", "")
        print("Test Set:", test_name)
        res_path = os.path.join(tmp_res_path, test_name)
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        # 拟合模型
        y_pred = predicter.predict(data=test_loader)
        # comp_dict = {'y_true': y_true, 'y_pred':y_pred, 'question_id':question_id}
        # torch.save(comp_dict, os.path.join(res_path, 'predict_{}.pt'.format(test_name)))
        y_true = targets
        acc_res_lit = compare_true_pre(y_true, y_pred, question_id)
        f_store_res.write("Test Set:{}\n".format(test_name))
        for v_ind, val in enumerate(acc_res_lit):
            f_store_res.write("{}".format(val))
            if v_ind != len(acc_res_lit) - 1:
                f_store_res.write(",")
            else:
                f_store_res.write("\n")
        # 计算题目的cls embedding
        # total_embedding = predicter.get_question_embedding(data=test_loader)
        # print(total_embedding.shape)
        # np.save(os.path.join("/home/cdk/python_project/bert_multi_tag_cls/dataset/my_raw/label_data_embdding/origin_bert",
        #                      test_name+'.npy'),
        #         total_embedding)

        # 释放显存
        if len(config['train']['n_gpu']) > 0:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    # cal_p_f_r()
    # according_index2question()