# Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks 

**Maksym Andriushchenko, Matthias Hein**

**University of Tübingen and Saarland University**

[http://arxiv.org/abs/1906.03526](http://arxiv.org/abs/1906.03526)

## Reproducible research
We provide code for **all** reported experiments with robust stumps and robust trees 
(`exps.sh` to train the models). 
Moreover, to foster reproducible research, we also provide code for **all** figures shown in the paper, 
each as a separate Jupyter notebook
(`toy2d.ipynb`, `model_analysis.ipynb`, `exact_adv.ipynb`). All dependencies are collected in `Dockerfile`.


## Main idea
We propose provable defenses against adversarial attack for boosted decision stumps and trees.
Here is the effect of our method on a 2D dataset.
<p align="center"><img src="images/toy2d_stumps_trees.png" width="900"></p>


## Provably robust training
We follow the framework of robust optimization aiming at solving the following min-max problem:
<p align="center"><img src="images/general_robust_optimization.png" width="300"></p>

We first derive the robustness certificates. The certification for boosted stumps is exact:
<p align="center"><img src="images/certificate_stumps.png" width="650"></p>
For boosted trees, we derive a simple lower bound on the functional margin which, however, 
becomes tight after robust training.
<p align="center"><img src="images/certificate_trees.png" width="550"></p>

Then we integrate these certificates into training which leads to the exact robust loss or to an upper bound on 
the robust loss for stumps and trees respectively.

How we minimize these robust losses? Surprisingly, it results in a convex optimization problem wrt the parameters of 
the stumps or trees. We use coordinate descent combined with bisection to solve for w_r and w_l. 
For more details, see the paper.


## Experiments
Experimental results show the efficiency of the robust training methods for boosted stumps and
boosted trees:
<p align="center"><img src="images/tables_rte.png" width="650"></p>


## Effect of robust training
The effect of robust training can be clearly seen based on the splitting thresholds 
that robust models select
<p align="center"><img src="images/thresholds_histograms.png" width="750"></p>


## Exact adversarial examples
Using our exact certification routine, we can also efficiently find provably minimal (exact) adversarial examples 
wrt Linf-norm for boosted stumps:
<p align="center"><img src="images/exact_adv_examples.png" width="800"></p>


## Code for training provably robust boosted stumps and trees
### Training
One can train robust stumps or trees using `train.py`.  The full list of possible arguments is 
available by `python train.py --help`. 

Boosted stumps models:
- `python train.py --dataset=mnist_2_6 --weak_learner=stump --model=plain `  
- `python train.py --dataset=mnist_2_6 --weak_learner=stump --model=robust_bound`
- `python train.py --dataset=mnist_2_6 --weak_learner=stump --model=robust_exact`

Boosted trees models:
- `python train.py --dataset=mnist_2_6 --weak_learner=tree --model=plain `  
- `python train.py --dataset=mnist_2_6 --weak_learner=tree --model=robust_bound`

Note that Linf epsilons for adversarial attacks are specified for each dataset separately in `data.py`.

### Evaluation
`eval.py` and `exact_adv.ipynb` show how one can restore a trained model in order to evaluate it (e.g., to
show the exact adversarial examples).

### Jupyter notebooks to reproduce the figures
- `toy2d.ipynb` - Figure 1: toy dataset which shows that the usual training is non-robust, while our robust models
can robustly classify all training points.
- `model_analysis.ipynb` - Figure 2: histograms of splitting thresholds, where we can observe a clear effect of 
robust training on the choice of the splitting thresholds.
- `exact_adv.ipynb` - Figure 3: exact adversarial examples for boosted stumps, 
which are much larger in Linf-norm for robust models. 

### Dependencies
All dependencies are collected in `Dockerfile`.
The best way to reproduce our environment is to use Docker. Just build the image and then run the container:
- `docker build -t provably_robust_boosting .`
- `docker run --name=boost -it -P -p 6001:6001 -t provably_robust_boosting`


## Contact
Please contact [Maksym Andriushchenko](https://github.com/max-andr) regarding this code.


## Citation
```
@article{andriushchenko2019provably,
  title={Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks},
  author={Andriushchenko, Maksym and Hein, Matthias},
  conference={arXiv preprint arXiv:1906.03526},
  year={2019}
}
```
