import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import SGD, lr_scheduler
from torch.autograd import Variable
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.util import Identity, AverageMeter, seed_torch, str2bool
from utils import ramps
# from models.resnet_3x3 import ResNet, BasicBlock
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
from utils.fair_evals import fair_test, cluster_acc

warnings.filterwarnings("ignore", category=UserWarning)

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

def warmup_train(model, train_loader, eva_loader, labeled_eval_loader, all_eval_loader, args):
    print("="*100)
    print("\t\t\t Warmup Training Start")
    print("="*100)
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.warmup_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(args.warmup_epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat = model(x, output='head2')
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Warmup_train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
    args.p_targets = target_distribution(probs)

def Baseline_train(model, train_loader, eva_loader, labeled_eval_loader, all_eval_loader, args):
    print("=" * 100)
    print("\t\t\t Baseline Training Start")
    print("=" * 100)
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat = model(x, output='head2')
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # wandb loss logging
        wandb.log({"loss/total_loss": loss_record.avg}, step=epoch)
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        DTC_acc, _, _, probs = test(model, eva_loader, args)

        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)

        # validation for unlabeled data with Backbone(on-the-fly) + head-2(on-the-fly)
        print('Head2: test on unlabeled classes')
        args.head = 'head2'
        acc_head2_ul, ind = fair_test(model, eva_loader, args, return_ind=True)

        # validation for labeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Joint Head1: test on labeled classes')
        args.head = 'head1'
        acc_head1_lb = fair_test(model, labeled_eval_loader, args, cluster=False)

        # validation for unlabeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Joint Head1: test on unlabeled classes')
        acc_head1_ul = fair_test(model, eva_loader, args, cluster=False, ind=ind)

        print('Joint Head1: test on all classes w/o clustering')
        acc_head1_all_wo_cluster = fair_test(model, all_eval_loader, args, cluster=False, ind=ind)

        print('Joint Head1: test on all classes w/ clustering')
        acc_head1_all_w_cluster = fair_test(model, all_eval_loader, args, cluster=True)

        # wandb metrics logging
        wandb.log({
            "val_acc/NCL_ul": DTC_acc,
            "val_acc/head2_ul": acc_head2_ul,
            "val_acc/head1_lb": acc_head1_lb,
            "val_acc/head1_ul": acc_head1_ul,
            "val_acc/head1_all_wo_clutering": acc_head1_all_wo_cluster,
            "val_acc/head1_all_w_clustering": acc_head1_all_w_cluster
        }, step=epoch)

    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


def PI_train(model, train_loader, eva_loader, labeled_eval_loader, all_eval_loader, args):
    print("=" * 100)
    print("\t\t\t PI Training Start")
    print("=" * 100)
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    w = 0
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)
            feat = model(x, output='head2')
            feat_bar = model(x_bar, output='head2')
            prob = feat2prob(feat, model.center)
            prob_bar = feat2prob(feat_bar, model.center)

            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)
            loss = sharp_loss + w * consistency_loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # wandb loss logging
        wandb.log({"loss/total_loss": loss_record.avg}, step=epoch)
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        DTC_acc, _, _, probs = test(model, eva_loader, args)
        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)

        # validation for unlabeled data with Backbone(on-the-fly) + head-2(on-the-fly)
        print('Head2: test on unlabeled classes')
        args.head = 'head2'
        acc_head2_ul, ind = fair_test(model, eva_loader, args, return_ind=True)

        # validation for labeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Joint Head1: test on labeled classes')
        args.head = 'head1'
        acc_head1_lb = fair_test(model, labeled_eval_loader, args, cluster=False)

        # validation for unlabeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Joint Head1: test on unlabeled classes')
        acc_head1_ul = fair_test(model, eva_loader, args, cluster=False, ind=ind)

        print('Joint Head1: test on all classes w/o clustering')
        acc_head1_all_wo_cluster = fair_test(model, all_eval_loader, args, cluster=False, ind=ind)

        print('Joint Head1: test on all classes w/ clustering')
        acc_head1_all_w_cluster = fair_test(model, all_eval_loader, args, cluster=True)

        # wandb metrics logging
        wandb.log({
            "val_acc/NCL_ul": DTC_acc,
            "val_acc/head2_ul": acc_head2_ul,
            "val_acc/head1_lb": acc_head1_lb,
            "val_acc/head1_ul": acc_head1_ul,
            "val_acc/head1_all_wo_clutering": acc_head1_all_wo_cluster,
            "val_acc/head1_all_w_clustering": acc_head1_all_w_cluster
        }, step=epoch)
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def test(model, test_loader, args):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_clusters))
    probs= np.zeros((len(test_loader.dataset), args.n_clusters))
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        feat = model(x, output='head2')
        prob = feat2prob(feat, model.center)
        _, pred = prob.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.cpu().detach().numpy()
        probs[idx, :] = prob.cpu().detach().numpy()
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = torch.from_numpy(probs)
    return acc, nmi, ari, probs

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
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--rampup_length', default=5, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=10.0)
    parser.add_argument('--milestones', default=[20, 40, 60, 80], type=int, nargs='+')
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
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='unitn-mhug')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= args.exp_root + '{}/{}'.format(runner_name, args.DTC)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'.pth'
    args.save_txt_path= args.exp_root+ '{}/{}/{}'.format(runner_name, args.DTC, args.save_txt_name)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    # WandB setting
    # use wandb logging
    wandb_run_name = args.model_name + '_NCL_supervised_' + str(args.seed)
    wandb.init(project='incd_dev_miu',
               entity=args.wandb_entity,
               name=wandb_run_name,
               mode=args.wandb_mode)

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
    model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
    save_weight_head1 = model.head1.weight.data.clone()
    save_bias_head1 = model.head1.bias.data.clone()

    model.head1 = Identity()
    init_feat_extractor = model
    init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor, unlabeled_eval_loader,
                                                                              args)
    args.p_targets = target_distribution(init_probs)

    model = ResNetDual(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
    model.load_state_dict(init_feat_extractor.state_dict(), strict=False)
    model.center= Parameter(torch.Tensor(args.n_clusters, args.n_clusters))
    model.center.data = torch.tensor(init_centers).float().to(device)

    model.head1.weight.data = save_weight_head1
    model.head1.bias.data = save_bias_head1

    # freeze head1
    frozen_layers = ['head1']
    freeze_layers(model, frozen_layers, True)

    warmup_train(model, train_loader, unlabeled_eval_loader, labeled_eval_loader, all_eval_loader, args)

    if args.DTC == 'Baseline':
        Baseline_train(model, train_loader, unlabeled_eval_loader, labeled_eval_loader, all_eval_loader, args)
    elif args.DTC == 'PI':
        PI_train(model, train_loader, unlabeled_eval_loader, labeled_eval_loader, all_eval_loader, args)



    acc, nmi, ari, _ = test(model, unlabeled_eval_loader, args)
    print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
    print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))
    if args.save_txt:
        with open(args.save_txt_path, 'a') as f:
            f.write("{:.4f}, {:.4f}, {:.4f}\n".format(acc, nmi, ari))

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
