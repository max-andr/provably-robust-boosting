#!/usr/bin/env bash

### Note: the best way to compare to our results is just to directly use the provided models.
### Some models retrained from scratch may give slightly different numbers.

# All datasets: breast_cancer diabetes cod_rna mnist_1_5 mnist_2_6 fmnist_sandal_sneaker gts_120_warning gts_30_70

# Train stumps on binary classification datasets
for model in plain robust_bound robust_exact; do
    nohup python train.py --dataset=breast_cancer         --weak_learner=stump --model=${model} --lr=1.0  >>  run_logs/breast_cancer-stump-${model}.out &
    nohup python train.py --dataset=diabetes              --weak_learner=stump --model=${model} --lr=1.0  >>  run_logs/diabetes-stump-${model}.out &
    nohup python train.py --dataset=cod_rna               --weak_learner=stump --model=${model} --lr=1.0  >>  run_logs/cod_rna-stump-${model}.out &
    nohup python train.py --dataset=mnist_1_5             --weak_learner=stump --model=${model} --lr=1.0  >>  run_logs/mnist_1_5-stump-${model}.out &
    nohup python train.py --dataset=mnist_2_6             --weak_learner=stump --model=${model} --lr=1.0  >>  run_logs/mnist_2_6-stump-${model}.out &
    nohup python train.py --dataset=fmnist_sandal_sneaker --weak_learner=stump --model=${model} --lr=1.0  >>  run_logs/fmnist_sandal_sneaker-stump-${model}.out &
    nohup python train.py --dataset=gts_100_roadworks     --weak_learner=stump --model=${model} --lr=1.0  >>  run_logs/gts_100_roadworks-stump-${model}.out &
    nohup python train.py --dataset=gts_30_70             --weak_learner=stump --model=${model} --lr=1.0  >>  run_logs/gts_30_70-stump-${model}.out &
done

# Train stumps with adversarial training (note: requires smaller learning rate)
for model in at_cube; do
    nohup python train.py --dataset=breast_cancer         --weak_learner=stump --model=${model} --lr=0.1  >>  run_logs/breast_cancer-stump-${model}.out &
    nohup python train.py --dataset=diabetes              --weak_learner=stump --model=${model} --lr=0.1  >>  run_logs/diabetes-stump-${model}.out &
    nohup python train.py --dataset=cod_rna               --weak_learner=stump --model=${model} --lr=0.1  >>  run_logs/cod_rna-stump-${model}.out &
    nohup python train.py --dataset=mnist_1_5             --weak_learner=stump --model=${model} --lr=0.1  >>  run_logs/mnist_1_5-stump-${model}.out &
    nohup python train.py --dataset=mnist_2_6             --weak_learner=stump --model=${model} --lr=0.1  >>  run_logs/mnist_2_6-stump-${model}.out &
    nohup python train.py --dataset=fmnist_sandal_sneaker --weak_learner=stump --model=${model} --lr=0.1  >>  run_logs/fmnist_sandal_sneaker-stump-${model}.out &
    nohup python train.py --dataset=gts_100_roadworks     --weak_learner=stump --model=${model} --lr=0.1  >>  run_logs/gts_100_roadworks-stump-${model}.out &
    nohup python train.py --dataset=gts_30_70             --weak_learner=stump --model=${model} --lr=0.1  >>  run_logs/gts_30_70-stump-${model}.out &
done

# Train trees on binary classification datasets
max_depth=4
for model in plain at_cube robust_bound; do
    nohup python train.py --dataset=breast_cancer         --weak_learner=tree --max_depth=$max_depth --model=${model} --lr=0.01 >>  run_logs/breast_cancer-tree-${model}.out &
    nohup python train.py --dataset=diabetes              --weak_learner=tree --max_depth=$max_depth --model=${model} --lr=0.2  >>  run_logs/diabetes-tree-${model}.out &
    nohup python train.py --dataset=cod_rna               --weak_learner=tree --max_depth=$max_depth --model=${model} --lr=0.2  >>  run_logs/cod_rna-tree-${model}.out &
    nohup python train.py --dataset=mnist_1_5             --weak_learner=tree --max_depth=$max_depth --model=${model} --lr=0.2  >>  run_logs/mnist_1_5-tree-${model}.out &
    nohup python train.py --dataset=mnist_2_6             --weak_learner=tree --max_depth=$max_depth --model=${model} --lr=0.2  >>  run_logs/mnist_2_6-tree-${model}.out &
    nohup python train.py --dataset=fmnist_sandal_sneaker --weak_learner=tree --max_depth=$max_depth --model=${model} --lr=0.2  >>  run_logs/fmnist_sandal_sneaker-tree-${model}.out &
    nohup python train.py --dataset=gts_100_roadworks     --weak_learner=tree --max_depth=$max_depth --model=${model} --lr=0.01 >>  run_logs/gts_100_roadworks-tree-${model}.out &
    nohup python train.py --dataset=gts_30_70             --weak_learner=tree --max_depth=$max_depth --model=${model} --lr=0.01 >>  run_logs/gts_30_70-tree-${model}.out &
done


### Multi-class experiments: robust models on MNIST, FMNIST, CIFAR-10
# Advice: multiclass models require quite some time to train. In case you want to get results faster you can try to
# subsample the thresholds by setting e.g. n_bins=10. However, this might slightly negatively affect the results.
nohup python train.py --dataset=mnist   --weak_learner=tree --max_depth=30 --model=robust_bound --lr=0.05  >>  run_logs/mnist-tree-robust_bound.out &
nohup python train.py --dataset=fmnist  --weak_learner=tree --max_depth=30 --model=robust_bound --lr=0.05  >>  run_logs/fmnist-tree-robust_bound.out &
nohup python train.py --dataset=cifar10 --weak_learner=tree --max_depth=4  --model=robust_bound --lr=0.1   >>  run_logs/cifar10-tree-robust_bound.out &

