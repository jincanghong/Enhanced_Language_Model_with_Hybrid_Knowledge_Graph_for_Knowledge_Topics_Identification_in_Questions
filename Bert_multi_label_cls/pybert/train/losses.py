#encoding:utf-8
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
import torch
from pybert.config.basic_config import configs
__call__ = ['CrossEntropy','BCEWithLogLoss']


class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss

gpu_use = configs['train']['n_gpu']


class BCEWithLogLoss(object):
    def __init__(self, use_cpu=False):
        ratio = 3
        if not use_cpu:
            pos_weight = (torch.ones([len(configs['my_label2id'])]) * ratio).to(torch.device('cuda:{}'.format(gpu_use[0])if
                                                                                         len(gpu_use) > 0 else 'cpu')) # ratio pos:neg = 3:1
        else:
            pos_weight = (torch.ones([len(configs['my_label2id'])]) * ratio)
        self.loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self, output, target):
        loss = self.loss_fn(input=output,target=target)
        return loss



# class my_loss(torch.nn.Module):
#     def __init__(self, use_cpu=False):
#         super(my_loss, self).__init__()
#         self.use_cpu = use_cpu
#     def forward(self, output, target):

