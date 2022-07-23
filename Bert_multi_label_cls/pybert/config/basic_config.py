#encoding:utf-8
from os import path
import json
import multiprocessing, json
from pathlib import Path
# BASE_DIR = Path('/home/cdk/python_project/bert_multi_tag_cls')
BASE_DIR = Path('pybert')
label2id = []
with open(BASE_DIR / 'dataset/all_data_tags/label2id.json','r', encoding='utf-8') as f:
    label2id = json.load(f)
    print(label2id)
# label2id = json.load(open(BASE_DIR / 'dataset/all_data_tags/label2id.json','r', encoding='utf-8')) # 名字->序号
bert_model_file = 'math_used_pretrain_model/kg_nsp_novelty_6w_pretrain_model_0724//'
# bert_model_file = 'pytorch_pretrain_chinese'
is_my = 'my_'
configs = {
    'task':'multi label',
    'data':{
        'raw_data_path': BASE_DIR / 'dataset/{}raw/train.csv'.format(is_my),  # 总的数据，一般是将train和test何在一起构建语料库
        'train_file_path': BASE_DIR / 'dataset/{}processed/train.tsv'.format(is_my),
        'valid_file_path': BASE_DIR / 'dataset/{}processed/valid.tsv'.format(is_my),
        'test_file_path': BASE_DIR / 'dataset/{}raw/tag_code_ai_all_clean.csv'.format(is_my),
        'id2name':BASE_DIR / 'dataset/{}raw/'.format(is_my),
        'test_length_dataset': False, # if is a list test set
        'test_length_dataset_path': BASE_DIR / 'dataset/{}raw/tag_code_ai_all.csv'.format(is_my)
    },
    'output':{
        'log_dir': BASE_DIR / 'output/log', # 模型运行日志
        'writer_dir': BASE_DIR / "output/TSboard",# TSboard信息保存路径
        'figure_dir': BASE_DIR / "output/figure", # 图形保存路径
        'checkpoint_dir': BASE_DIR / "output/checkpoints/07_24_kg_nsp_novelty_math_6w_pretrain",
        # 'checkpoint_dir': BASE_DIR / "output/checkpoints/07_22_unpretrain",
        'result': BASE_DIR / "output/result",  # test result
        'cache_dir': BASE_DIR / 'model/',
    },
    'pretrained':{
        "bert":{
            'vocab_path_eng': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/vocab.txt',
            'tf_checkpoint_path': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/bert_model.ckpt',
            'vocab_path_ch': BASE_DIR / 'model/{}/bert-base-chinese-vocab.txt'.format(bert_model_file),
            'bert_model_dir': BASE_DIR / 'model/{}'.format(bert_model_file),
            'bert_config_file': BASE_DIR / 'model/pretrain/{}/bert_config.json'.format(bert_model_file),
            'pytorch_model_path': BASE_DIR / 'model/{}/pytorch_model.bin'.format(bert_model_file),
        },
        'embedding':{}
    },

    # 5e-4 LR, 1e-4 WD better 30epoch will convergency; 5e-4, 2e-4  40epoch will
    'train':{
        'valid_size': 0,
        'max_seq_len': 256,
        'do_lower_case': False,
        'batch_size': 480,  # 24,  # how many samples to process at once
        'epochs': 250,  # number of epochs to train
        'start_epoch': 1,
        'warmup_proportion': 0.1,# Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
        'gradient_accumulation_steps': 6, # Number of updates steps to accumulate before performing a backward/update pass.
        'learning_rate': 5e-4, #Adam 学习率
        'n_gpu': [], # GPU个数,如果只写一个数字，则表示gpu标号从0开始，并且默认使用gpu:0作为controller,
                       # 如果以列表形式表示，即[1,3,5],则我们默认list[0]作为controller
        'num_workers': multiprocessing.cpu_count(),  # 线程个数
        'weight_decay': 2e-4, #权重衰减
        'seed':2018,  #随机种子 42
        'resume': False,
        # 'resume':'epoch_50_0.0137_bert_model.pth',
    },
    'predict':{
        'batch_size': 1024
    },
    'callbacks':{
        'lr_patience': 3, # number of epochs with no improvement after which learning rate will be reduced.
        'mode': 'min',    # one of {min, max}
        'monitor': 'valid_loss',  # 计算指标
        'early_patience': 20,   # early_stopping
        'save_best_only': False, # 是否保存最好模型
        'save_checkpoint_freq': 5  # 保存模型频率，当save_best_only为False时候，指定才有作用
    },
    'my_label2id': label2id,
    'model':{
        'arch':'bert'
    },
    'id_match_entity_path': '/home/cdk/python_project/bert_multi_tag_cls/dataset/pretrain_data/id_match_entity_6w.json',
    'entity2id_path' : '/home/cdk/python_project/bert_multi_tag_cls/clustering_tfidf/math_kg/entity2id.txt'
}