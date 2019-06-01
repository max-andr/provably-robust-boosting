#!/usr/bin/env bash

# All datasets: breast_cancer diabetes cod_rna mnist_2_6 fmnist_sandal_sneaker gts_120_warning gts_30_70

weak_learner=stump
for dataset in breast_cancer diabetes cod_rna mnist_2_6 fmnist_sandal_sneaker gts_100_roadworks gts_30_70; do
    nohup python train.py --dataset=$dataset --weak_learner=$weak_learner --model=plain        >> run_logs/${dataset}-${weak_learner}-plain.out &
    nohup python train.py --dataset=$dataset --weak_learner=$weak_learner --model=robust_bound >> run_logs/${dataset}-${weak_learner}-robust_bound.out &
    nohup python train.py --dataset=$dataset --weak_learner=$weak_learner --model=robust_exact >> run_logs/${dataset}-${weak_learner}-robust_exact.out &
done

weak_learner=tree
for dataset in breast_cancer diabetes cod_rna mnist_2_6 fmnist_sandal_sneaker gts_100_roadworks gts_30_70; do
    nohup python train.py --dataset=$dataset --weak_learner=$weak_learner --model=plain        >> run_logs/${dataset}-${weak_learner}-plain.out &
    nohup python train.py --dataset=$dataset --weak_learner=$weak_learner --model=robust_bound >> run_logs/${dataset}-${weak_learner}-robust_bound.out &
done
