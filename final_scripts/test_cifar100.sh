#!/usr/bin/env bash

python -W ignore fianl_DTC_cifar100.py \
          --DTC PI \
          --seed 5 \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 30 \
          --epochs 200 \
          --batch_size 128 \
          --rampup_length 5 \
          --rampup_coefficient 10.0 \
          --update_interval 10 \
          --weight_decay 1e-5 \
          --save_txt true \
          --dataset_root ./data/datasets/CIFAR/ \
          --exp_root ./data/experiments/ \
          --model_name DTC_PI_cifar100 \
          --save_txt_name PI_results_cifar100.txt \
          --num_unlabeled_classes 20 \
          --num_labeled_classes 80 \
          --n_clusters 20 \
          --dataset_name cifar100 \
          --pretrain_dir ./data/experiments/supervised_learning_wo_ssl/warmup_cifar100_resnet_wo_ssl.pth