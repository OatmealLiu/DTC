#!/usr/bin/env bash

python -W ignore DTC_incd_train_tinyimagenet.py \
          --DTC PI \
          --seed 5 \
          --warmup_lr 0.1 \
          --lr 0.05 \
          --warmup_epochs 1 \
          --epochs 1 \
          --batch_size 128 \
          --rampup_length 5 \
          --rampup_coefficient 10.0 \
          --gamma 0.5 \
          --update_interval 10 \
          --weight_decay 1e-5 \
          --milestones 20 40 60 80 \
          --save_txt true \
          --dataset_root ./data/datasets/tiny-imagenet-200/ \
          --exp_root ./data/experiments/ \
          --model_name DTC_PI_tinyimagenet \
          --save_txt_name PI_results_tinyimagenet.txt \
          --num_unlabeled_classes 20 \
          --num_labeled_classes 180 \
          --n_clusters 20 \
          --dataset_name tinyimagenet \
          --pretrain_dir ./data/experiments/supervised_learning_wo_ssl/warmup_TinyImageNet_resnet_wo_ssl.pth \
          --wandb_mode online \
          --wandb_entity oatmealliu