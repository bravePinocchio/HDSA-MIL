import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils.utils import initialize_weights
import numpy as np
from models.resnet_custom import resnet50_baseline
import os

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.2))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x

class Fusion_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False):
        super(Fusion_Net, self).__init__()
        self.fc_a = [nn.Linear(L, D), nn.ReLU(), nn.Dropout(0.25)]
        self.fc_b = [nn.Linear(640, D), nn.ReLU(), nn.Dropout(0.25)]
        self.fc_a = nn.Sequential(*self.fc_a)
        self.fc_b = nn.Sequential(*self.fc_b)
        self.fc = [nn.Linear(D*2, D), nn.Sigmoid()]
        self.fc = nn.Sequential(*self.fc)
    def forward(self, a, b):
        a = self.fc_a(a)  ## 1024 -> 256
        b = self.fc_b(b.to(torch.float32))  ## 576 -> 256
        c = torch.cat([a, b], dim=1)
        return c

class My_model(nn.Module):
    def __init__(self, gate=True, dropout=False, k_sample=8, n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(),
                 subtyping=False, training_control = False, FilterInst = False, smoothE = 0, filter_num=0, gamma = 0):
        super(My_model, self).__init__()
        size = [1024, 512, 256]
        #fc = [nn.Linear(size[0], size[1]), nn.Linear(size[1], size[1]), nn.ReLU()]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifiers = nn.Linear(size[1] * n_classes, n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.k_sample, self.n_classes  = k_sample, n_classes
        self.instance_loss_fn = instance_loss_fn
        self.subtyping, self.FilterInst = subtyping, FilterInst

        self.smoothE, self.filter_num, self.gamma = smoothE, filter_num, gamma

        self.training_control = training_control
        self.first_label = None
        self.batch_already = False
        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_sample = self.k_sample
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]

        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(len(top_p), device)
        n_targets = self.create_negative_targets(len(top_n), device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_sample = self.k_sample
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(len(top_p), device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def get_already(self, label):
        if self.first_label is not None:
            if not self.batch_already:
                if label != self.first_label:
                    self.batch_already = True
        else:
            self.first_label = label

    def get_bag_pred(self, A, h, label):
        attn_score = A[int(label)]
        if '_Top' in self.FilterInst:
            idx_to_keep = torch.topk(-attn_score, k= len(h) - self.filter_num )[1]
        elif '_ThreProb' in self.FilterInst:
            # strategy B: remove the positive instance above prob K
            idx_to_keep = torch.where(attn_score <= int(self.FilterInst.split('_ThreProb')[-1]) / 100.0)[0]
            if idx_to_keep.shape[0] == 0:  # if all instance are dropped, keep the most positive one
                idx_to_keep = torch.topk(attn_score, k=1)[1]
        feat_removedNeg = h[idx_to_keep]  ## [N', 512]
        attn_removeNeg = torch.transpose(A, 0, 1)[idx_to_keep]
        attn_removeNeg = torch.transpose(attn_removeNeg, 0, 1)
        M = torch.mm(attn_removeNeg, feat_removedNeg)
        return M

    def get_smooth_attn(self, A):
        #B = (1-A)
        C = (1-A).pow(self.gamma)
        B = C * A
        return B

    def forward(self, feat, label=None, instance_eval=False, return_features=False,
                attention_only=False, cur=None, training=False, epoch = 0):
        device = feat.device
        if self.training_control:
            self.get_already(label)
        else:
            self.batch_already = True

        A, h = self.attention_net(feat)
        A = torch.transpose(A, 1, 0)

        if attention_only:    return A
        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]  # W[inst，m]
                if inst_label == 1:
                    instance_loss, preds, targets= self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping and self.batch_already:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss
            if self.subtyping and self.batch_already:
                total_inst_loss /= len(self.instance_classifiers)

        if training:
            if self.FilterInst:
                if epoch >= self.smoothE:
                    B = self.get_smooth_attn(A)
                    M = torch.mm(B, h)
                    #M = self.get_bag_pred(A, h, label)
                else:
                    M = torch.mm(A, h)
            else:
                M = torch.mm(A, h)
        else:
            M = torch.mm(A, h)

        M = M.view(1, -1)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class sb_model(My_model):
    def __init__(self, gate=True, dropout=False, k_sample=8, n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(),
                 subtyping=False, training_control = False, FilterInst = None, smoothE = 0, filter_num=0, gamma=0):
        nn.Module.__init__(self)
        size = [1024, 512, 256]
        #fc = [nn.Linear(size[0], size[1]), nn.Linear(size[1], size[1]), nn.ReLU()]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.k_sample, self.n_classes = k_sample, n_classes
        self.instance_loss_fn = instance_loss_fn
        self.FilterInst, self.subtyping = FilterInst, subtyping

        self.smoothE, self.filter_num, self.gamma =  smoothE, filter_num, gamma

        self.training_control = training_control
        self.first_label = None
        self.batch_already = False
        initialize_weights(self)

    def forward(self, feat, label=None,  instance_eval=False, return_features=False,
                attention_only=False, cur=None, training=False, epoch = 0):
        device = feat.device
        if self.training_control:
            self.get_already(label)
        else:
            self.batch_already = True

        A, h = self.attention_net(feat)
        A = torch.transpose(A, 1, 0)

        if attention_only:  return A
        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping and self.batch_already:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping and self.batch_already:
                total_inst_loss /= len(self.instance_classifiers)

        if training:
            if self.FilterInst:
                if epoch >= self.smoothE:
                    B = self.get_smooth_attn(A)
                    M = torch.mm(B,h)

                else:
                    M = torch.mm(A, h)
            else:
                M = torch.mm(A, h)
        else:
            M = torch.mm(A, h)

        M = M.view(1, -1)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits, Y_prob, Y_hat, A_raw, results_dict

class clam_model(My_model):
    def __init__(self, gate=True, dropout=False, k_sample=8, n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(),
                 subtyping=False, training_control = False, FilterInst = None, smoothE = 0, filter_num=0, gamma=0):
        nn.Module.__init__(self)
        size = [1024, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.k_sample, self.n_classes = k_sample, n_classes
        self.instance_loss_fn = instance_loss_fn
        self.FilterInst, self.subtyping = FilterInst, subtyping

        self.smoothE, self.filter_num, self.gamma =  smoothE, filter_num, gamma

        self.training_control = training_control
        self.first_label = None
        self.batch_already = False
        initialize_weights(self)

    def forward(self, feat, label=None,  instance_eval=False, return_features=False,
                attention_only=False, cur=None, training=False, epoch = 0):
        device = feat.device
        if self.training_control:
            self.get_already(label)
        else:
            self.batch_already = True

        A, h = self.attention_net(feat)
        A = torch.transpose(A, 1, 0)

        if attention_only:  return A
        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping and self.batch_already:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping and self.batch_already:
                total_inst_loss /= len(self.instance_classifiers)

        if training:
            if self.FilterInst:
                if epoch >= self.smoothE:
                    B = self.get_smooth_attn(A)
                    M = torch.mm(B,h)
                else:
                    M = torch.mm(A, h)
            else:
                M = torch.mm(A, h)
        else:
            M = torch.mm(A, h)

        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits, Y_prob, Y_hat, A_raw, results_dict

if __name__ == '__main__':
    label = torch.tensor([1,1,1,0,1,0,0]).cuda()
    data = torch.randn([7,45,1024]).cuda()
    model = My_model(dropout=True,subtyping=True,training_control=True)
    model = model.cuda()
    for i in range(len(data)):
        out = model(feat=data[i],label=label[i], instance_eval=True)
    print(model)