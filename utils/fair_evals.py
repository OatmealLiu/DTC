from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
# from sklearn.utils.linear_assignment_ import linear_assignment # LOOK I DON'T HAVE THIS VERSION
import scipy.io
from tqdm import tqdm
import random
import os
import argparse
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from modules.module import feat2prob, target_distribution
# from modules.module import feat2prob, target_distribution

#######################################################
# Evaluate Critiron
#######################################################
# def cluster_acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     ind = linear_assignment(w.max() - w)
#
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #ind = linear_assignment(w.max() - w)    # DEBUG: original version, return a tuple of ndarraies represent idx, jdx
                                             #        it can't be iterated as the method below in a for-loop
    # LOOK: modified version of code
    ind_arr, jnd_arr = linear_assignment(w.max() - w)
    ind = np.array(list(zip(ind_arr, jnd_arr)))

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def fair_test(model, test_loader, args, cluster=True, ind=None, return_ind=False):
    model.eval()
    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)
        if args.step == 'first':
            output1, output2, _ = model(x, output='both')
            if args.head == 'head1':
                output = torch.cat((output1, output2), dim=1)
            else:
                output = feat2prob(output2, model.center)
        else:
            output1, output2, output3, _ = model(x, output='all')
            if args.head == 'head1':
                output = torch.cat((output1, output2, output3), dim=1)
            elif args.head == 'head2':
                output = feat2prob(output2, model.center)
            elif args.head == 'head3':
                output = feat2prob(output3, model.center)

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    if cluster:
        if return_ind:
            acc, ind = cluster_acc(targets.astype(int), preds.astype(int), return_ind)
        else:
            acc = cluster_acc(targets.astype(int), preds.astype(int), return_ind)
        nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)
        print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    else:
        if ind is not None:
            if args.step == 'first':
                ind = ind[:args.num_unlabeled_classes1, :]
                idx = np.argsort(ind[:, 1])
                id_map = ind[idx, 0]
                id_map += args.num_labeled_classes

                # targets_new = targets
                targets_new = np.copy(targets)
                for i in range(args.num_unlabeled_classes1):
                    targets_new[targets == i + args.num_labeled_classes] = id_map[i]
                targets = targets_new
            else:
                ind = ind[:args.num_unlabeled_classes2, :]
                idx = np.argsort(ind[:, 1])
                id_map = ind[idx, 0]
                id_map += args.num_labeled_classes

                # targets_new = targets
                targets_new = np.copy(targets)
                for i in range(args.num_unlabeled_classes2):
                    targets_new[targets == i + args.num_labeled_classes] = id_map[i]
                targets = targets_new

        preds = torch.from_numpy(preds)
        targets = torch.from_numpy(targets)
        correct = preds.eq(targets).float().sum(0)
        acc = float(correct / targets.size(0))
        print('Test acc {:.4f}'.format(acc))

    if return_ind:
        return acc, ind
    else:
        return acc
