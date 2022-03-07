#!/usr/bin/env bash

python -W ignore two_DTC_incd_train_cifar100.py \
          --DTC PI \
          --seed 5 \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 30 \
          --epochs 200 \
          --batch_size 128 \
          --rampup_length 5 \
          --rampup_coefficient 10.0 \
          --gamma 0.5 \
          --update_interval 10 \
          --weight_decay 1e-5 \
          --milestones 20 40 60 80 \
          --save_txt true \
          --dataset_root ./data/datasets/CIFAR/ \
          --exp_root ./data/experiments/ \
          --model_name DTC_1st_PI_cifar100 \
          --save_txt_name PI_1st_results_cifar100.txt \
          --num_unlabeled_classes1 10 \
          --num_unlabeled_classes2 10 \
          --num_labeled_classes 80 \
          --n_clusters 10 \
          --dataset_name cifar100 \
          --pretrain_dir ./data/experiments/supervised_learning_wo_ssl/warmup_cifar100_resnet_wo_ssl.pth \
          --wandb_mode online \
          --wandb_entity oatmealliu \
          --step first

python -W ignore two_DTC_incd_train_cifar100.py \
          --DTC PI \
          --seed 5 \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 30 \
          --epochs 200 \
          --batch_size 128 \
          --rampup_length 5 \
          --rampup_coefficient 10.0 \
          --gamma 0.5 \
          --update_interval 10 \
          --weight_decay 1e-5 \
          --milestones 20 40 60 80 \
          --save_txt true \
          --dataset_root ./data/datasets/CIFAR/ \
          --exp_root ./data/experiments/ \
          --model_name DTC_2nd_PI_cifar100 \
          --save_txt_name PI_2nd_results_cifar100.txt \
          --num_unlabeled_classes1 10 \
          --num_unlabeled_classes2 10 \
          --num_labeled_classes 80 \
          --n_clusters 10 \
          --dataset_name cifar100 \
          --pretrain_dir ./data/experiments/supervised_learning_wo_ssl/warmup_cifar100_resnet_wo_ssl.pth \
          --wandb_mode online \
          --wandb_entity oatmealliu \
          --step second \
          --first_step_dir ./data/experiments/two_DTC_incd_train_cifar100/PI/first_DTC_1st_PI_cifar100.pth
