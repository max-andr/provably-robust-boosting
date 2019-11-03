import numpy as np
from tree_ensemble import Tree


class OneVsAllClassifier:
    def __init__(self, models):
        self.models = models
        self.n_clsf = len(models)

    def predict(self, X):
        preds = np.zeros([self.n_clsf, X.shape[0]])
        for i_cls in range(self.n_clsf):
            preds[i_cls] = self.models[i_cls].predict(X)
        return preds

    def predict_class(self, X):
        preds = self.predict(X)
        if self.n_clsf == 1:
            return (0.5*((preds > 0)+1)).astype(int).flatten()
        else:
            return np.argmax(preds, 0)

    def certify_treewise(self, X, y, eps):
        preds = np.zeros([self.n_clsf, X.shape[0]])
        for i_cls in range(self.n_clsf):
            preds[i_cls] = self.models[i_cls].certify_treewise(X, y[i_cls], eps)
        return preds

    def certify_exact(self, X, y, eps):
        preds = np.zeros([self.n_clsf, X.shape[0]])
        for i_cls in range(self.n_clsf):
            preds[i_cls] = self.models[i_cls].certify_exact(X, y[i_cls], eps)
        return preds

    def fmargin(self, X, y, fx_vals=None):
        if fx_vals is None:  # if fx_vals have not been provided
            fx_vals = self.predict(X)
        if self.n_clsf > 1:
            preds_correct_class = (fx_vals * (y == 1)).sum(0, keepdims=True)
            diff = preds_correct_class - fx_vals  # difference between the correct class and all other classes
            diff[y == 1] = np.inf  # to exclude zeros coming from f_correct - f_correct
            fx_vals = diff.min(0, keepdims=True)
        else:
            fx_vals = y * fx_vals
        return fx_vals[0]

    def fmargin_treewise(self, X, y, eps, fx_vals=None):
        if fx_vals is None:  # if fx_vals have not been provided
            fx_vals = self.certify_treewise(X, y, eps)
        if self.n_clsf > 1:
            cert_correct_class = (fx_vals * (y == 1)).sum(0, keepdims=True)
            diff = cert_correct_class + fx_vals  # plus because of [min -f] in cert for all classes
            fx_vals = np.min(diff, 0, keepdims=True)
        return fx_vals[0]

    def fmargin_exact(self, X, y, eps):
        fx_vals = self.certify_exact(X, y, eps)
        if self.n_clsf > 1:
            cert_correct_class = (fx_vals * (y == 1)).sum(0, keepdims=True)
            diff = cert_correct_class + fx_vals  # plus because of [min -f] in cert for all classes
            fx_vals = np.min(diff, 0, keepdims=True)
        return fx_vals[0]

    def save(self, model_path):
        if model_path != '':
            model_lst = []
            for model in self.models:
                model_lst.append(model.export_model())
            model_arr = np.array(model_lst)
            np.save(model_path, model_arr)

    def load(self, model_path, iteration=-1):
        model_data = np.load(model_path, allow_pickle=True)
        for i_clsf in range(self.n_clsf):
            self.models[i_clsf].load(model_data[i_clsf], iteration)
        if type(model_data[0]) is dict:
            n_trees = max(model_data[0].keys()) + 1
        else:
            n_trees = model_data.shape[1]
        true_iteration = iteration + 1 if iteration != -1 else n_trees
        print('Ensemble of {}/{} trees restored: {}'.format(true_iteration, n_trees, model_path))

    def dump_model(self):
        """ Returns the model in JSON format compatible with XGBoost. """
        # Works for trees
        n_cls = len(self.models)
        n_trees = max([len(model.trees) for model in self.models])

        list_of_tree_dicts = []
        for i_tree in range(n_trees):
            for i_cls in range(n_cls):
                if i_tree < len(self.models[i_cls].trees):
                    tree = self.models[i_cls].trees[i_tree]
                else:
                    tree = Tree()
                tree_dict, _ = tree.get_json_dict(counter_terminal_nodes=-10)
                list_of_tree_dicts.append(tree_dict)

        return list_of_tree_dicts
