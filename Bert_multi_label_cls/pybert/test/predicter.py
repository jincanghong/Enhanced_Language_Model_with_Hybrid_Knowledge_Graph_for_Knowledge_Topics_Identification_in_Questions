#encoding:utf-8
import torch
import numpy as np
from ..utils.utils import model_device, load_bert
import torch.nn.functional as F


class Predicter(object):
    def __init__(self,
                 model,
                 logger,
                 n_gpu,
                 model_path
                 ):
        self.model = model
        self.logger = logger
        self.width = 30
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model, logger=self.logger)
        loads = load_bert(model_path=model_path,model = self.model)
        self.model = loads[0]

    def show_info(self,batch_id,n_batch):
        recv_per = int(100 * (batch_id + 1) / n_batch)
        if recv_per >= 100:
            recv_per = 100
        # 进度条模式
        show_bar = f"\r[predict]{batch_id+1}/{n_batch}[{int(self.width * recv_per / 100) * '>':<{self.width}s}]{recv_per}%"
        print(show_bar,end='')

    def predict(self,data):
        all_logits = 0
        self.model.eval()
        n_batch = len(data)
        with torch.no_grad():
            for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(data):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                # print("input_ids:{}".format(input_ids))
                logits = self.model(input_ids, segment_ids, input_mask)
                # logits = F.softmax(logits, dim=1)
                logits = logits.sigmoid()
                self.show_info(step,n_batch)
                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                else:
                    if step == 0:
                        # print(torch.argsort(-logits)[:10])
                        all_logits = logits.detach().cpu().numpy()
                    else:
                        all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
        return all_logits

    def get_question_embedding(self, data):
        self.model.eval()
        n_batch = len(data)
        total_embedding = None
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                cls_embedding = self.model(input_ids, segment_ids, input_mask, get_cls=True)
                self.show_info(step, n_batch)
                if total_embedding is None:
                    total_embedding = cls_embedding.detach().cpu().numpy()
                else:
                    total_embedding = np.concatenate([total_embedding, cls_embedding.detach().cpu().numpy()], axis=0)
        return total_embedding


