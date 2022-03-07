#!/usr/bin/env bash

python -W ignore fianl_DTC_cifar10.py \
          --DTC PI \
          --seed 5 \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 10 \
          --epochs 200 \
          --batch_size 128 \
          --rampup_length 5 \
          --rampup_coefficient 10.0 \
          --update_interval 5 \
          --weight_decay 1e-4 \
          --save_txt true \
          --dataset_root ./data/datasets/CIFAR/ \
          --exp_root ./data/experiments/ \
          --model_name DTC_PI_cifar10 \
          --save_txt_name PI_results_cifar10.txt \
          --num_unlabeled_classes 5 \
          --num_labeled_classes 5 \
          --n_clusters 5 \
          --dataset_name cifar10 \
          --pretrain_dir ./data/experiments/supervised_learning_wo_ssl/warmup_cifar10_resnet_wo_ssl.pth

