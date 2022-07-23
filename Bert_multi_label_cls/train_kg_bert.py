# -*- coding: utf-8 -*-
# __author__ = 'K_HOLMES_'
# __time__   = '2019/7/21 16:26'

#encoding:utf-8
import torch
import warnings, os,math, numpy as np

from math_bert.math_dataset import CreateDataset
from math_bert.model import BertMath
from math_bert.kg_trainer import Trainer

from pybert.train.losses import BCEWithLogLoss
from torch.utils.data import DataLoader
from pybert.io_data.data_transformer import DataTransformer
from pybert.utils.logginger import init_logger
from pybert.utils.utils import seed_everything, text_write
from pybert.config.basic_config import configs as config
from pybert.callback.lrscheduler import BertLR
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import F1Score,AccuracyThresh,MultiLabelReport
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
warnings.filterwarnings("ignore")


# 主函数
def main():
    # **************************** 基础信息 ***********************
    logger = init_logger(log_name=config['model']['arch'], log_dir=config['output']['log_dir'])
    logger.info(f"seed is {config['train']['seed']}")
    device = f"cuda: {config['train']['n_gpu'][0] if len(config['train']['n_gpu']) else 'cpu'}"
    seed_everything(seed=config['train']['seed'],device=device)
    logger.info('starting load data from disk')
    id2label = {value: key for key, value in config['my_label2id'].items()}
    id_match_entity_path = '/home/cdk/python_project/bert_multi_tag_cls/dataset/pretrain_data/id_match_entity_6w.json'
    entity2id_path = '/home/cdk/python_project/bert_multi_tag_cls/clustering_tfidf/math_kg/entity2id.txt'
    # **************************** 数据生成 ***********************
    DT = DataTransformer(logger=logger,seed=config['train']['seed'])
    # 读取数据集以及数据划分
    writable = False
    if writable or not (os.path.exists(config['data']['train_file_path']) and os.path.exists(config['data']['valid_file_path'])):
        targets, sentences, question_ids = DT.read_data(raw_data_path=config['data']['raw_data_path'],
                                            is_train=True)
        if config['train']['valid_size'] != 0:
            train, valid = DT.train_val_split(X=sentences, y=targets, question_ids=question_ids, save=True, shuffle=True,
                                              valid_size=config['train']['valid_size'],
                                              train_path=config['data']['train_file_path'],
                                              valid_path=config['data']['valid_file_path'])
        else:
            # 将test部分作为valid, train不划分valid
            train = list(zip(question_ids, sentences, targets))
            # shuffle(train)
            print("Train size:{}".format(len(train)))
            text_write(config['data']['train_file_path'], train, np=False)
            val_tar, val_sen, question_ids = DT.read_data(raw_data_path=config['data']['test_file_path'],
                                            is_train=True)
            size = len(val_sen) // 15
            valid = list(zip(question_ids, val_sen, val_tar))  # 作为验证集
            valid = valid[:size]
            print("Validation size:{}".format(len(valid)))
            text_write(config['data']['valid_file_path'], valid, np=False)

    else:
        train = DT.read_processed_data(config['data']['train_file_path'])
        valid = DT.read_processed_data(config['data']['valid_file_path'])

    tokenizer = BertTokenizer(vocab_file=config['pretrained']['bert']['vocab_path_ch'],
                              do_lower_case=config['train']['do_lower_case'])

    config['train']['batch_size'] = config['train']['batch_size'] // config['train']['gradient_accumulation_steps']
    # train
    train_dataset  = CreateDataset( data=train,
                                    tokenizer=tokenizer,
                                    max_seq_len=config['train']['max_seq_len'],
                                    seed=config['train']['seed'],
                                    example_type='train',
                                    id_match_entity_path=config['id_match_entity_path'],
                                    entity2id_path=config['entity2id_path']
                                   )
    # valid
    valid_dataset = CreateDataset(  data= valid,
                                    tokenizer = tokenizer,
                                    max_seq_len  = config['train']['max_seq_len'],
                                    seed = config['train']['seed'],
                                    example_type = 'valid',
                                    id_match_entity_path=config['id_match_entity_path'],
                                    entity2id_path=config['entity2id_path']
                                  )
    #加载训练数据集
    train_loader = DataLoader(dataset     = train_dataset,
                              batch_size  = config['train']['batch_size'],
                              num_workers = config['train']['num_workers'],
                              shuffle     = True,
                              drop_last   = False,
                              pin_memory  = False)
    # 验证数据集
    valid_loader = DataLoader(dataset     = valid_dataset,
                              batch_size  = config['train']['batch_size'],
                              num_workers = config['train']['num_workers'],
                              shuffle     = False,
                              drop_last   = False,
                              pin_memory  = False)

    # **************************** 模型 ***********************
    logger.info("initializing model")
    model = BertMath.from_pretrained(config['pretrained']['bert']['bert_model_dir'],
                                     cache_dir=config['output']['cache_dir'],
                                     num_classes=len(id2label))

    # ************************** 优化器 *************************
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    num_train_steps = int(
        len(train_dataset.examples) / config['train']['batch_size'] / config['train']['gradient_accumulation_steps'] * config['train']['epochs'])
    # t_total: total number of training steps for the learning rate schedule
    # warmup: portion of t_total for the warmup
    print("t_total:{}".format(num_train_steps))
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config['train']['learning_rate'],
                         warmup=config['train']['warmup_proportion'],
                         t_total=num_train_steps)

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")
    # 模型保存
    model_checkpoint = ModelCheckpoint(checkpoint_dir   = config['output']['checkpoint_dir'],
                                       mode             = config['callbacks']['mode'],
                                       monitor          = config['callbacks']['monitor'],
                                       save_best_only   = config['callbacks']['save_best_only'],
                                       arch             = config['model']['arch'],
                                       epoch_freq       = config['callbacks']['save_checkpoint_freq'],
                                       logger           = logger)
    # 监控训练过程
    train_monitor = TrainingMonitor(file_dir = config['output']['figure_dir'],
                                    arch = config['model']['arch'])
    # 学习率机制
    lr_scheduler = BertLR(optimizer = optimizer,
                          learning_rate = config['train']['learning_rate'],
                          t_total = num_train_steps,
                          warmup = config['train']['warmup_proportion'])

    # **************************** training model ***********************
    logger.info('training model....')
    vecs = []
    vecs.append([0] * 100)
    with open("clustering_tfidf/math_kg/entity2vec.vec", 'r') as fin:
        for ind, line in enumerate(fin):
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed = torch.FloatTensor(vecs)
    train_configs = {
        'model': model,
        'logger': logger,
        'optimizer': optimizer,
        'resume': config['train']['resume'],
        'epochs': config['train']['epochs'],
        'n_gpu': config['train']['n_gpu'],
        'gradient_accumulation_steps': config['train']['gradient_accumulation_steps'],
        'epoch_metrics':[F1Score(average='micro', task_type='binary', search_thresh=True),MultiLabelReport(id2label=id2label)],
        'batch_metrics':[AccuracyThresh(thresh=0.5)],
        'criterion': BCEWithLogLoss(),
        'criterion_cpu': BCEWithLogLoss(use_cpu=True),
        'model_checkpoint': model_checkpoint,
        'training_monitor': train_monitor,
        'lr_scheduler': lr_scheduler,
        'early_stopping': None,
        'verbose': 1,
        'embed':embed
    }

    trainer = Trainer(train_configs=train_configs)
    # 拟合模型
    trainer.train(train_data=train_loader,valid_data=valid_loader)
    # 释放显存
    if len(config['train']['n_gpu']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()