import numpy as np
from tqdm.auto import tqdm

from ParTree.algorithms.bic_estimator import bic
from ParTree.algorithms.data_splitter import DecisionSplit
from ParTree.classes.ParTree import ParTree


class VarianceParTree(ParTree):

    def __init__(self,
                 max_depth=3,
                 max_nbr_clusters=10,
                 min_samples_leaf=3,
                 min_samples_split=5,
                 max_nbr_values=100,
                 max_nbr_values_cat=10,
                 bic_eps=0.0,
                 random_state=None,
                 n_jobs: int = 1,
                 verbose=False
                 ):
        super().__init__(
            max_depth,
            max_nbr_clusters,
            min_samples_leaf,
            min_samples_split,
            max_nbr_values,
            max_nbr_values_cat,
            bic_eps,
            random_state,
            n_jobs,
            verbose
        )

    def _make_split(self, idx_iter):
        n_features = self.X.shape[1]

        best_feature = None
        best_threshold = None
        best_var = np.inf

        results = []

        for n in tqdm(range(n_features), position=0, leave=False, disable=not self.verbose):
            for feature in tqdm(self.feature_values[n], position=1, leave=False, disable=not self.verbose):
                #X, is_categorical_feature, idx_iter, feature, threshold
                results.append(self.processPoolExecutor.submit(_make_split_innerloop,
                                                               self.X,
                                                               self.is_categorical_feature,
                                                               idx_iter,
                                                               n,
                                                               feature))

        for res in tqdm(results, disable=not self.verbose):
            best_returned_mse, best_returned_feature, best_returned_threshold = res.result()

            if best_returned_mse < best_var:
                best_feature = best_returned_feature
                best_threshold = best_returned_threshold
                best_var = best_returned_mse

        if best_feature is None:
            return None, np.zeros(len(self.X[idx_iter])), np.inf

        clf = DecisionSplit(best_feature, best_threshold, self.is_categorical_feature[best_feature])
        labels = clf.apply(self.X[idx_iter])
        bic_children = bic(self.X[idx_iter], (np.array(labels) - 1).tolist())
        is_oblique = False

        return clf, labels, bic_children, is_oblique

def _make_split_innerloop(X, is_categorical_feature, idx_iter, feature, threshold):

    if not is_categorical_feature[feature]:  # splitting feature is continuous
        cond = X[idx_iter, feature] <= threshold
        X_a = X[idx_iter][cond]
        X_b = X[idx_iter][~cond]
    else:  # splitting feature is categorical
        cond = X[idx_iter, feature] == threshold
        X_a = X[idx_iter][cond]
        X_b = X[idx_iter][~cond]

    if len(X_a) == 0 or len(X_b) == 0:
        return [np.inf, feature, threshold]
        #raise ValueError(f"VarianceParTree: len(X_a) = {len(X_a)} or len(X_b) = {len(X_b)}")

    var_a = np.var(X_a)  # np.mean(np.var(X_a, axis=0))
    var_b = np.var(X_b)  # np.mean(np.var(X_b, axis=0))

    var_tot = len(X_a) / len(X[idx_iter]) * var_a + len(X_b) / len(X[idx_iter]) * var_b

    return [var_tot, feature, threshold]
