import random

import torch
from torch.optim import Adam

from data import IntentDset
from model import ProtNet
from torch import nn, optim
import numpy as np
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")


# from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear


# https://github.com/cyvius96/prototypical-network-pytorch/blob/master/utils.py
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def get_evaluate(preds, y):
    # 四舍五入到最接近的整数
    pres = preds.clone().detach().cpu()
    rounded_preds = np.asarray(pres).round()
    val_predict = np.argmax(rounded_preds, axis=1)
    acc = accuracy_score(y.cpu(), val_predict)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y.cpu(), val_predict, average="macro")
    fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), pres.numpy()[:, 1], pos_label=1)
    val_auc = metrics.auc(fpr, tpr)
    return acc, val_precision, val_recall, val_f1, val_auc


Nc = 2
Ni = 5
Nq = 16
# 源域
# charliehebdo ferguson germanwings-carsh ottawashooting sydneysiege
dataset = ['ferguson', 'sydneysiege', 'germanwings-crash', 'ottawashooting']
train_set = []
for i in dataset:
    train_set.append(IntentDset(dataset=i, n_query=Nq, Nc=Nc, Ni=Ni))
# 目标域
val_dset = IntentDset(dataset='charliehebdo', Nc=Nc, n_query=Nq, Ni=Ni)

pn = ProtNet().cuda(1)

param_optimizer = list(pn.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = Adam(optimizer_grouped_parameters, lr=5e-5)

criterion = nn.CrossEntropyLoss()

step = 0

while True:
    pn.train()
    step += 1
    # print('gpu_usage',round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    batch = random.sample(train_set, 1)[0].next_batch()
    sup_set = batch['sup_set_x']
    qry_set = batch['target_x']

    # https://discuss.pytorch.org/t/multiple-model-forward-followed-by-one-loss-backward/20868/2
    # two forwards will link to two different instance wont overwrite the model
    sup = pn(sup_set['input_ids'].cuda(1), sup_set['input_mask'].cuda(1))
    qry = pn(qry_set['input_ids'].cuda(1), qry_set['input_mask'].cuda(1))

    # sup = sup.view(Ni, Nc, -1).mean(0)
    s1 = torch.mean(sup[:Ni], axis=0)
    s2 = torch.mean(sup[Ni:], axis=0)
    sup = torch.tensor([item.cpu().detach().numpy() for item in [s1, s2]]).cuda(1)
    logits = euclidean_metric(qry, sup)

    label = torch.tensor(qry_set['label']).cuda(1)
    # label = torch.arange(Nc).repeat(Nq).type(torch.LongTensor).cuda(1)

    loss = criterion(logits, label)

    # print('gpu_usage',round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # if step % 1 == 0:
    #     print('Iteration :', step, "Loss :", float(loss.item()))

    if step % 125 == 0:
        pn.eval()
        pn.cuda(1)
        acc_list = []
        val_precision_list = []
        val_recall_list = []
        val_f1_all_list = []
        val_auc_all_list = []
        for i in range(100):
            batch = val_dset.next_batch()
            sup_set = batch['sup_set_x']
            qry_set = batch['target_x']

            sup = pn(sup_set['input_ids'].cuda(1), sup_set['input_mask'].cuda(1))
            qry = pn(qry_set['input_ids'].cuda(1), qry_set['input_mask'].cuda(1))

            s1 = torch.mean(sup[:Ni], axis=0)
            s2 = torch.mean(sup[Ni:], axis=0)
            sup = torch.tensor([item.cpu().detach().numpy() for item in [s1, s2]]).cuda(1)
            # sup = sup.view(Ni, Nc, -1).mean(0)
            # logits = euclidean_metric(qry, sup).max(1)[1].cpu()
            logits = euclidean_metric(qry, sup)

            label = torch.tensor(qry_set['label']).cuda(1)

            # label = torch.arange(Nc).repeat(Nq).type(torch.LongTensor)

            acc, val_precision, val_recall, val_f1, val_auc = get_evaluate(logits, label)
            acc_list.append(acc)
            val_precision_list.append(val_precision)
            val_recall_list.append(val_recall)
            val_f1_all_list.append(val_f1)
            val_auc_all_list.append(val_auc)
        print(f'Valid Acc: {np.mean(acc_list) * 100:.2f}')
        print(
            f'\tValid Recall: {np.mean(val_recall_list) * 100:.2f}| '
            f'Valid Precision: {np.mean(val_precision_list) * 100:.2f}| Valid f1: {np.mean(val_f1_all_list) * 100:.2f}| Valid Auc: '
            f'{np.mean(val_auc_all_list) * 100:.2f}')
        print("-------------------------------")
    if step % 100000 == 0:
        break
