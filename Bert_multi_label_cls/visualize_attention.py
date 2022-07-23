# -*- coding: utf-8 -*-
# __author__ = 'K_HOLMES_'
# __time__   = '2019/7/1 17:24'

from pytorch_pretrained_bert.modeling import BertModel
from math_bert.novelty_modeling import BertModel as MathBert
import torch, collections, os
root_dir = '/home/cdk/python_project/bert_multi_tag_cls/model'
checkpoints_path = '/home/cdk/python_project/bert_multi_tag_cls/output/checkpoints/best_model_checkpoints/epoch_60_0.0167_bert_model.pth'
pretrained_model_path = os.path.join(root_dir, 'math_used_pretrain_model/kg_nsp_novelty_6w_pretrain_model_0724')
pretrained_model = MathBert.from_pretrained(pretrained_model_path)

pretrained_para_list = list(pretrained_model.named_parameters())
pretrained_para_list = [each[0] for each in pretrained_para_list]

trained_model_para = torch.load(checkpoints_path, lambda storage, loc: storage)
trained_model_para = trained_model_para['state_dict']

parameter_list = list(trained_model_para.keys())
drop_list = parameter_list[-2:]
for each in drop_list:
    trained_model_para.pop(each)

assert len(trained_model_para.keys()) == len(pretrained_para_list)  # 两个要相等

tmp_ordered_dicts = collections.OrderedDict()
for ind, name in enumerate(trained_model_para.keys()): # 得到新的model的值
    tmp_ordered_dicts[pretrained_para_list[ind]] = trained_model_para.get(name)

pretrained_model.load_state_dict(tmp_ordered_dicts)
print("加载新model参数完成")
model_to_save = pretrained_model.module if hasattr(pretrained_model, 'module') else pretrained_model
torch.save(model_to_save.state_dict(), os.path.join(root_dir, 'my_model/pytorch_model.bin'))
print("存储完成")