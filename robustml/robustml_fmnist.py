import sys
sys.path.append('..')
import robustml
from classifiers import OneVsAllClassifier
from tree_ensemble import TreeEnsemble


def load_model(model_path):
    n_classifiers, weak_learner, ensembles = 10, 'tree', []

    for i_clsf in range(n_classifiers):
        # hyperparameters are not important when loading
        ensembles.append(TreeEnsemble(weak_learner, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    model_ova = OneVsAllClassifier(ensembles)
    model_ova.load(model_path)
    return model_ova


class Model(robustml.model.Model):
    def __init__(self, sess):
        model_path = "models/models_trees_multiclass/2019-08-06 14:59:51 dataset=fmnist weak_learner=tree model=robust_bound n_train=-1 n_trials_coord=784 eps=0.100 max_depth=30 lr=0.05.model.npy"
        self.model = load_model(model_path)

        self._dataset = robustml.dataset.FMNIST()
        self._threat_model = robustml.threat_model.Linf(epsilon=0.1)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        predictions = self.model.predict(x)
        pred_label = predictions.argmax()  # label as a number
        return pred_label

