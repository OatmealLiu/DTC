import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import SGD
from torch.autograd import Variable
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.util import Identity, AverageMeter, seed_torch, str2bool
from utils import ramps
from models.resnet import ResNetDual, BasicBlock
from modules.module import feat2prob, target_distribution
from data.cifarloader import CIFAR10Loader, CIFAR10LoaderMix, CIFAR100Loader, CIFAR100LoaderMix
from data.tinyimagenetloader import TinyImageNetLoader
from tqdm import tqdm
import numpy as np
import warnings
import random
import os
import wandb
from collections.abc import Iterable
from utils.fair_evals import cluster_acc

def init_prob_kmeans(model, eval_loader, args):
    torch.manual_seed(1)
    model = model.to(device)
    # cluster parameter initiate
    model.eval()
    targets = np.zeros(len(eval_loader.dataset))
    feats = np.zeros((len(eval_loader.dataset), 512))
    for _, (x, label, idx) in enumerate(eval_loader):
        x = x.to(device)
        feat = model(x, output='head1')
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.data.cpu().numpy()
        targets[idx] = label.data.cpu().numpy()
    # evaluate clustering performance
    pca = PCA(n_components=args.n_clusters)
    feats = pca.fit_transform(feats)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(feats)
    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = feat2prob(torch.from_numpy(feats), torch.from_numpy(kmeans.cluster_centers_))
    return acc, nmi, ari, kmeans.cluster_centers_, probs

def fair_test(model, test_loader, args, cluster=True, ind=None, return_ind=False):
    model.eval()
    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)
        output1, output2, _ = model(x, output='both')
        output2 = feat2prob(output2, model.center)
        if args.head == 'head1':
            output = torch.cat([output1, output2], dim=1)
        else:
            output = output2

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
            ind = ind[:args.num_unlabeled_classes, :]
            idx = np.argsort(ind[:, 1])
            id_map = ind[idx, 0]
            id_map += args.num_labeled_classes

            # targets_new = targets
            targets_new = np.copy(targets)
            for i in range(args.num_unlabeled_classes):
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

def freeze_layers(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def unfreeze_layers(model, layer_names):
    freeze_layers(model, layer_names, False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--warmup_lr', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--rampup_length', default=5, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=10.0)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--update_interval', default=5, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, tinyimagenet')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_txt', default=False, type=str2bool, help='save txt or not', metavar='BOOL')
    parser.add_argument('--pretrain_dir', type=str, default='./data/experiments/pretrained/resnet18_cifar10_classif_5.pth')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--save_txt_name', type=str, default='result.txt')
    parser.add_argument('--DTC', type=str, default='PI')
    parser.add_argument('--wandb_mode', type=str, default='offline', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='oatmealliu')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = "DTC_incd_train_cifar10"
    model_dir= args.exp_root + '{}/{}'.format(runner_name, args.DTC)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'.pth'
    args.save_txt_path= args.exp_root+ '{}/{}/{}'.format(runner_name, args.DTC, args.save_txt_name)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if args.dataset_name == 'cifar10':
        # train_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
    elif args.dataset_name == 'cifar100':
        # train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
    elif args.dataset_name == 'tinyimagenet':
        # train_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug='twice', shuffle=True, class_list = range(args.num_labeled_classes, num_classes), subfolder='train')
        train_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug='twice', shuffle=True, class_list = range(args.num_labeled_classes, num_classes), subfolder='train')
        unlabeled_eval_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, class_list = range(args.num_labeled_classes, num_classes), subfolder='train')
        unlabeled_eval_loader_test = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, class_list = range(args.num_labeled_classes, num_classes), subfolder='val')
        labeled_eval_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, class_list = range(args.num_labeled_classes), subfolder='val')
        all_eval_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, class_list = range(num_classes), subfolder='val')



    model = ResNetDual(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
    state_dict = torch.load(args.model_dir)
    model.load_state_dict(state_dict, strict=False)

    model.head1 = Identity()
    init_feat_extractor = model
    init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor,
                                                                              unlabeled_eval_loader,
                                                                              args)
    args.p_targets = target_distribution(init_probs)

    model = ResNetDual(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
    state_dict = torch.load(args.model_dir)
    model.load_state_dict(state_dict, strict=False)
    model.center = Parameter(torch.Tensor(args.n_clusters, args.n_clusters))
    model.center.data = torch.tensor(init_centers).float().to(device)

    model.eval()
# =============================== Final Test ===============================
    acc_list = []

    print('Head2: test on unlabeled classes')
    args.head = 'head2'
    _, ind = fair_test(model, unlabeled_eval_loader, args, return_ind=True)

    print('Evaluating on Head1')
    args.head = 'head1'

    print('test on labeled classes (test split)')
    acc = fair_test(model, labeled_eval_loader, args, cluster=False)
    acc_list.append(acc)

    print('test on unlabeled classes (test split)')
    acc = fair_test(model, unlabeled_eval_loader_test, args, cluster=False, ind=ind)
    acc_list.append(acc)

    print('test on all classes w/o clustering (test split)')
    acc = fair_test(model, all_eval_loader, args, cluster=False, ind=ind)
    acc_list.append(acc)

    print('test on all classes w/ clustering (test split)')
    acc = fair_test(model, all_eval_loader, args, cluster=True)
    acc_list.append(acc)

    print('Evaluating on Head2')
    args.head = 'head2'

    print('test on unlabeled classes (train split)')
    acc = fair_test(model, unlabeled_eval_loader, args)
    acc_list.append(acc)

    print('test on unlabeled classes (test split)')
    acc = fair_test(model, unlabeled_eval_loader_test, args)
    acc_list.append(acc)

    print('Acc List: Joint Head1->Old, New, All_wo_cluster, All_w_cluster, Head2->Train, Test')
    print(acc_list)
