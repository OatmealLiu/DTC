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
from models.resnet import ResNetDual, BasicBlock, ResNetTri
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

def PI_train(model, train_loader, eva_loader, labeled_eval_loader, all_eval_loader, args):
    print("="*100)
    print("\t\t\t\t\t1st-step Training")
    print("="*100)
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

        # LOOK: Let's use our evaluation to test their model
        # validation for unlabeled data with Backbone(on-the-fly) + head-2(on-the-fly)
        print('Head2: test on unlabeled classes')
        args.head = 'head2'
        acc_head2_ul, ind = fair_test(model, eva_loader, args, return_ind=True)

        # validation for labeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Head1: test on labeled classes')
        args.head = 'head1'
        acc_head1_lb = fair_test(model, labeled_eval_loader, args, cluster=False)

        # validation for unlabeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Head1: test on unlabeled classes')
        acc_head1_ul = fair_test(model, eva_loader, args, cluster=False, ind=ind)

        # validation for all
        print('Head1: test on all classes w/o clustering')
        acc_head1_all_wo_cluster = fair_test(model, all_eval_loader, args, cluster=False, ind=ind)

        print('Head1: test on all classes w/ clustering')
        acc_head1_all_w_cluster = fair_test(model, all_eval_loader, args, cluster=True)

        # wandb metrics logging
        wandb.log({
            "val_acc/head2_ul": acc_head2_ul,
            "val_acc/head1_lb": acc_head1_lb,
            "val_acc/head1_ul": acc_head1_ul,
            "val_acc/head1_all_wo_clutering": acc_head1_all_wo_cluster,
            "val_acc/head1_all_w_clustering": acc_head1_all_w_cluster
        }, step=epoch)
        # LOOK: our method ends

    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def warmup_train_second(model, train_loader, eva_loader, labeled_eval_loader, all_eval_loader, args):
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
            feat = model(x, output='head3')
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Warmup_train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
    args.p_targets = target_distribution(probs)

def PI_train_second(model, train_loader, eva_loader, labeled_eval_loader, all_eval_loader, p_unlabeled_eval_loader, args):
    print("="*100)
    print("\t\t\t\t\t2nd-step Training")
    print("="*100)

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
            feat = model(x, output='head3')
            feat_bar = model(x_bar, output='head3')
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
        if epoch % args.update_interval == 0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)

        # LOOK: Let's use our evaluation to test their model
        # validation for unlabeled data with Backbone(on-the-fly) + head-2(on-the-fly)
        args.head = 'head2'
        print('Head2: test on PRE-unlabeled classes')
        acc_head2_ul, ind2 = fair_test(model, p_unlabeled_eval_loader, args, return_ind=True)

        args.head = 'head3'
        print('Head3: test on unlabeled classes')
        acc_head3_ul, ind3 = fair_test(model, eva_loader, args, return_ind=True)

        # validation for labeled data with Backbone(on-the-fly) + head-1(frozen)
        args.head = 'head1'
        print('Head1: test on labeled classes')
        acc_head1_lb = fair_test(model, labeled_eval_loader, args, cluster=False)

        # validation for unlabeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Head1: test on PRE-unlabeled classes')
        acc_head1_ul1 = fair_test(model, p_unlabeled_eval_loader, args, cluster=False, ind=ind2)

        # validation for unlabeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Head1: test on unlabeled classes')
        acc_head1_ul2 = fair_test(model, eva_loader, args, cluster=False, ind=ind3)

        # validation for all
        print('Head1: test on all classes w/o clustering')
        acc_head1_all_wo_cluster = fair_test(model, all_eval_loader, args, cluster=False, ind=ind3)

        print('Head1: test on all classes w/ clustering')
        acc_head1_all_w_cluster = fair_test(model, all_eval_loader, args, cluster=True)

        # wandb metrics logging
        wandb.log({
            "val_acc/head2_ul": acc_head2_ul,
            "val_acc/head3_ul": acc_head3_ul,
            "val_acc/head1_lb": acc_head1_lb,
            "val_acc/head1_ul_1": acc_head1_ul1,
            "val_acc/head1_ul_2": acc_head1_ul2,
            "val_acc/head1_all_wo_clutering": acc_head1_all_wo_cluster,
            "val_acc/head1_all_w_clustering": acc_head1_all_w_cluster
        }, step=epoch)
        # LOOK: our method ends

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
        if args.step == 'first':
            feat = model(x, output='head2')
        else:
            feat = model(x, output='head3')
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
    parser.add_argument('--num_unlabeled_classes1', default=10, type=int)
    parser.add_argument('--num_unlabeled_classes2', default=10, type=int)
    parser.add_argument('--num_labeled_classes', default=80, type=int)
    parser.add_argument('--n_clusters', default=10, type=int)
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
    parser.add_argument('--step', type=str, default='first', choices=['first', 'second'])
    parser.add_argument('--first_step_dir', type=str,
                        default='./results/two_incd_cifar100_DTC/DTC_cifar100_incd_resnet18_80.pth')
    parser.add_argument('--second_step_dir', type=str,
                        default='./results/two_incd_cifar100_DTC/DTC_cifar100_incd_resnet18_80.pth')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = args.exp_root + '{}/{}'.format(runner_name, args.DTC)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir +'/' + args.step + '_' + args.model_name+'.pth'
    args.save_txt_path = args.exp_root+ '{}/{}/{}'.format(runner_name, args.DTC, args.save_txt_name)

    # WandB setting
    # use wandb logging
    wandb_run_name = args.model_name + '_DTC_' + str(args.seed)
    wandb.init(project='incd_dev_miu',
               entity=args.wandb_entity,
               name=wandb_run_name,
               mode=args.wandb_mode)

    if args.DTC == 'PI' and args.step == 'first':
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes1
        unlabeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                                aug='twice', shuffle=True,
                                                target_list=range(args.num_labeled_classes, num_classes))
        unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None,
                                              shuffle=False, target_list=range(args.num_labeled_classes, num_classes))
        unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                               aug=None,
                                               shuffle=False, target_list=range(args.num_labeled_classes, num_classes))
        labeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                             shuffle=False, target_list=range(args.num_labeled_classes))
        all_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(num_classes))

        model = ResNetDual(BasicBlock, [2,2,2,2], args.num_labeled_classes,
                           args.num_unlabeled_classes1+args.num_unlabeled_classes2).to(device)
        model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
        model.head2 = nn.Linear(512, args.num_unlabeled_classes1).to(device)

        save_weight_head1 = model.head1.weight.data.clone()
        save_bias_head1 = model.head1.bias.data.clone()

        model.head1 = Identity()
        init_feat_extractor = model
        init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor, unlabeled_val_loader,
                                                                                  args)
        args.p_targets = target_distribution(init_probs)

        model = ResNetDual(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes1).to(device)
        model.load_state_dict(init_feat_extractor.state_dict(), strict=False)
        model.center = Parameter(torch.Tensor(args.n_clusters, args.n_clusters))
        model.center.data = torch.tensor(init_centers).float().to(device)

        model.head1.weight.data = save_weight_head1
        model.head1.bias.data = save_bias_head1

        # freeze head1
        frozen_layers = ['head1']
        freeze_layers(model, frozen_layers, True)

        warmup_train(model, unlabeled_train_loader, unlabeled_val_loader, labeled_test_loader, all_test_loader, args)

        PI_train(model, unlabeled_train_loader, unlabeled_val_loader, labeled_test_loader, all_test_loader, args)

        # LOOK: OUR TEST FLOW
        # =============================== Final Test ===============================
        acc_list = []

        print('Head2: test on unlabeled classes')
        args.head = 'head2'
        _, ind = fair_test(model, unlabeled_val_loader, args, return_ind=True)

        print('Evaluating on Head1')
        args.head = 'head1'

        print('test on labeled classes (test split)')
        acc = fair_test(model, labeled_test_loader, args, cluster=False)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model, unlabeled_test_loader, args, cluster=False, ind=ind)
        acc_list.append(acc)

        print('test on all classes w/o clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=False, ind=ind)
        acc_list.append(acc)

        print('test on all classes w/ clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=True)
        acc_list.append(acc)

        print('Evaluating on Head2')
        args.head = 'head2'

        print('test on unlabeled classes (train split)')
        acc = fair_test(model, unlabeled_val_loader, args)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model, unlabeled_test_loader, args)
        acc_list.append(acc)

        print('Acc List: Synthesized Head1->Old, New, All_wo_cluster, All_w_cluster, Head2->Train, Test')
        print(acc_list)
    elif args.DTC == 'PI' and args.step == 'second':
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes1 + args.num_unlabeled_classes2
        unlabeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                                aug='twice', shuffle=True,
                                                target_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                  num_classes))
        unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None, shuffle=False,
                                              target_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                num_classes))
        unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                               aug=None, shuffle=False,
                                               target_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                 num_classes))
        labeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                             shuffle=False, target_list=range(args.num_labeled_classes))
        all_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(num_classes))

        # Previous step Novel classes dataloader
        p_unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                                aug=None, shuffle=False,
                                                target_list=range(args.num_labeled_classes,
                                                                  args.num_labeled_classes + args.num_unlabeled_classes1))
        p_unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                                 aug=None, shuffle=False,
                                                 target_list=range(args.num_labeled_classes,
                                                                   args.num_labeled_classes + args.num_unlabeled_classes1))

        model = ResNetDual(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                           args.num_unlabeled_classes1 + args.num_unlabeled_classes2).to(device)
        model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
        model.head2 = nn.Linear(512, args.num_unlabeled_classes1).to(device)

        model.head1 = Identity()
        init_feat_extractor = model
        init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor, unlabeled_val_loader,
                                                                                  args)
        args.p_targets = target_distribution(init_probs)

        model = ResNetTri(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes1,
                          args.num_unlabeled_classes2).to(device)

        model.load_state_dict(torch.load(args.first_step_dir), strict=False)
        model.center = Parameter(torch.Tensor(args.n_clusters, args.n_clusters))
        model.center.data = torch.tensor(init_centers).float().to(device)

        # freeze head1
        frozen_layers = ['head1']
        freeze_layers(model, frozen_layers, True)

        warmup_train_second(model, unlabeled_train_loader, unlabeled_val_loader, labeled_test_loader,
                            all_test_loader, args)
        PI_train_second(model, unlabeled_train_loader, unlabeled_val_loader, labeled_test_loader, all_test_loader,
                        p_unlabeled_val_loader, args)


        # LOOK: OUR TEST FLOW
        # =============================== Final Test ===============================
        acc_list = []

        args.head = 'head2'
        print('Head2: test on unlabeled classes')
        _, ind1 = fair_test(model, p_unlabeled_val_loader, args, return_ind=True)

        args.head = 'head3'
        print('Head3: test on unlabeled classes')
        _, ind2 = fair_test(model, unlabeled_val_loader, args, return_ind=True)

        args.head = 'head1'
        print('Evaluating on Head1')

        print('test on labeled classes (test split)')
        acc = fair_test(model, labeled_test_loader, args, cluster=False)
        acc_list.append(acc)

        print('test on unlabeled classes 1nd-NEW (test split)')
        acc = fair_test(model, p_unlabeled_test_loader, args, cluster=False, ind=ind1)
        acc_list.append(acc)

        print('test on unlabeled classes 2nd-NEW (test split)')
        acc = fair_test(model, unlabeled_test_loader, args, cluster=False, ind=ind2)
        acc_list.append(acc)

        print('test on all classes w/o clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=False, ind=ind2)
        acc_list.append(acc)

        print('test on all classes w/ clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=True)
        acc_list.append(acc)

        args.head = 'head2'
        print('Evaluating on Head2')

        print('test on unlabeled classes (train split)')
        acc = fair_test(model, p_unlabeled_val_loader, args)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model, p_unlabeled_test_loader, args)
        acc_list.append(acc)

        args.head = 'head3'
        print('Evaluating on Head3')

        print('test on unlabeled classes (train split)')
        acc = fair_test(model, unlabeled_val_loader, args)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model, unlabeled_test_loader, args)
        acc_list.append(acc)

        print(
            'Acc List: Head1 -> Old, New-1, New-2, All_wo_cluster, All_w_cluster, Head2->Train, Test, Head3->Train, Test')
        print(acc_list)
    elif args.DTC == 'eval' and args.step == 'second':
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes1 + args.num_unlabeled_classes2
        unlabeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                                aug='twice', shuffle=True,
                                                target_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                  num_classes))
        unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None, shuffle=False,
                                              target_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                num_classes))
        unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                               aug=None, shuffle=False,
                                               target_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                 num_classes))
        labeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                             shuffle=False, target_list=range(args.num_labeled_classes))
        all_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(num_classes))

        # Previous step Novel classes dataloader
        p_unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                                aug=None, shuffle=False,
                                                target_list=range(args.num_labeled_classes,
                                                                  args.num_labeled_classes + args.num_unlabeled_classes1))
        p_unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                                 aug=None, shuffle=False,
                                                 target_list=range(args.num_labeled_classes,
                                                                   args.num_labeled_classes + args.num_unlabeled_classes1))

        model_new1 = ResNetDual(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                           args.num_unlabeled_classes1 + args.num_unlabeled_classes2).to(device)
        model_new1.head2 = nn.Linear(512, args.num_unlabeled_classes1).to(device)
        model_new1.load_state_dict(torch.load(args.first_step_dir), strict=False)

        center_model1 = ResNetDual(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                                   args.num_unlabeled_classes1 + args.num_unlabeled_classes2).to(device)
        center_model1.head2 = nn.Linear(512, args.num_unlabeled_classes1).to(device)
        center_model1.load_state_dict(torch.load(args.first_step_dir), strict=False)

        center_model1.head1 = Identity()
        init_feat_extractor = center_model1
        init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor,
                                                                                  p_unlabeled_val_loader,
                                                                                  args)
        args.p_targets = target_distribution(init_probs)

        model_new1.center = Parameter(torch.Tensor(args.n_clusters, args.n_clusters))
        model_new1.center.data = torch.tensor(init_centers).float().to(device)

        # model new 2
        model_new2 = ResNetTri(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes1,
                          args.num_unlabeled_classes2).to(device)
        model_new2.load_state_dict(torch.load(args.second_step_dir), strict=False)

        center_model2 = ResNetTri(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes1,
                                  args.num_unlabeled_classes2).to(device)
        center_model2.load_state_dict(torch.load(args.second_step_dir), strict=False)

        center_model2.head1 = Identity()
        init_feat_extractor2 = center_model2
        init_acc2, init_nmi2, init_ari2, init_centers2, init_probs2 = init_prob_kmeans(init_feat_extractor2,
                                                                                  p_unlabeled_val_loader,
                                                                                  args)

        model_new2.center = Parameter(torch.Tensor(args.n_clusters, args.n_clusters))
        model_new2.center.data = torch.tensor(init_centers).float().to(device)

        # LOOK: OUR TEST FLOW
        # =============================== Final Test ===============================
        acc_list = []

        args.head = 'head2'
        args.step = 'first'
        print('Head2: test on unlabeled classes')
        _, ind1 = fair_test(model_new1, p_unlabeled_val_loader, args, return_ind=True)

        args.step = 'second'
        args.head = 'head3'
        print('Head3: test on unlabeled classes')
        _, ind2 = fair_test(model_new2, unlabeled_val_loader, args, return_ind=True)

        args.head = 'head1'
        print('Evaluating on Head1')
        acc_all = 0.
        print('test on labeled classes (test split)')
        acc = fair_test(model_new2, labeled_test_loader, args, cluster=False)
        acc_list.append(acc)
        acc_all = acc_all + acc * args.num_labeled_classes

        args.step_model = 'new1'
        print('test on unlabeled classes 1nd-NEW (test split)')
        acc = fair_test(model_new2, p_unlabeled_test_loader, args, cluster=False, ind=ind1)
        acc_list.append(acc)
        acc_all = acc_all + acc * args.num_unlabeled_classes1

        args.step_model = 'new2'
        print('test on unlabeled classes 2nd-NEW (test split)')
        acc = fair_test(model_new2, unlabeled_test_loader, args, cluster=False, ind=ind2)
        acc_list.append(acc)
        acc_all = acc_all + acc * args.num_unlabeled_classes2

        print('test on all classes w/o clustering (test split)')
        acc = acc_all / num_classes
        acc_list.append(acc)

        print('test on all classes w/ clustering (test split)')
        acc = acc_all / num_classes
        acc_list.append(acc)

        args.head = 'head2'
        print('Evaluating on Head2')

        print('test on unlabeled classes (train split)')
        acc = fair_test(model_new2, p_unlabeled_val_loader, args)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model_new2, p_unlabeled_test_loader, args)
        acc_list.append(acc)

        args.head = 'head3'
        print('Evaluating on Head3')

        print('test on unlabeled classes (train split)')
        acc = fair_test(model_new2, unlabeled_val_loader, args)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model_new2, unlabeled_test_loader, args)
        acc_list.append(acc)

        print(
            'Acc List: Head1 -> Old, New-1, New-2, All_wo_cluster, All_w_cluster, Head2->Train, Test, Head3->Train, Test')
        print(acc_list)

