import numpy as np
from itertools import repeat

from scipy import stats
from scipy.spatial.distance import cdist
from scipy.spatial.distance import seuclidean, jaccard

from ParTree.algorithms.bic_estimator import bic
from ParTree.classes.ParTree import ParTree
from ParTree.algorithms.data_splitter import DecisionSplit
from tqdm.auto import tqdm


class CenterParTree(ParTree):
    def __init__(
            self,
            max_depth=3,
            max_nbr_clusters=10,
            min_samples_leaf=3,
            min_samples_split=5,
            max_nbr_values=100,
            max_nbr_values_cat=10,
            bic_eps=0.0,
            random_state=None,
            metric="euclidean",
            n_jobs=1,
            verbose = False
    ):
        """
        For continuous features the algorithm uses "metric" distance, for categorical ones the Mode, and for mixed types
        it uses Seuclidean distance and Jaccard distance.

        :param metric:
            The distance metric to use. For more information see the cdist metric argument in the scipy documentation
        """
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
        self.metric = metric

    def _make_split(self, idx_iter):
        n_features = self.X.shape[1]

        best_feature = None
        best_threshold = None
        best_mse = np.inf

        for res in tqdm(self.processPoolExecutor.map(_make_split_innerloop,
                                                    repeat(self.X),
                                                    repeat(self.con_indexes),
                                                    repeat(self.cat_indexes),
                                                    repeat(self.metric),
                                                    repeat(self.is_categorical_feature),
                                                    repeat(idx_iter),
                                                    range(n_features),
                                                    repeat(self.feature_values),
                                                     repeat(self.verbose)),
                        disable=not self.verbose, position=0, leave=False):

            best_returned_mse, best_returned_feature, best_returned_threshold = res

            if best_returned_mse < best_mse:
                best_feature = best_returned_feature
                best_threshold = best_returned_threshold
                best_mse = best_returned_mse

        if best_feature is None:
            return None, np.zeros(len(self.X[idx_iter])), np.inf, False

        clf = DecisionSplit(best_feature, best_threshold, self.is_categorical_feature[best_feature])

        labels = clf.apply(self.X[idx_iter])
        bic_children = bic(self.X[idx_iter], (np.array(labels) - 1).tolist())

        return clf, labels, bic_children, False


def _mixed_metric(con_indexes, cat_indexes, u, v):
    con_dist = seuclidean(u[con_indexes], v[con_indexes], V=np.ones(len(con_indexes)))
    cat_dist = jaccard(u[cat_indexes], v[cat_indexes])
    con_w = len(con_indexes) / len(u)
    cat_w = len(cat_indexes) / len(u)
    dist = con_w * con_dist + cat_w * cat_dist
    return dist


def _make_split_innerloop(X, con_indexes, cat_indexes, metric, is_categorical_feature, idx_iter, feature,
                          feature_values, verbose, X_perc=.2, min_X=1000):
    best_feature = None
    best_mse = np.inf
    best_threshold = None

    n_features = X.shape[1]
    for feature in tqdm(self.feature_values[n], position=1, leave=False, disable=not verbose):

        cond = X[idx_iter, feature] == threshold if is_categorical_feature[feature] \
            else X[idx_iter, feature] <= threshold

        X_a = X[idx_iter][cond]
        X_a = X_a[np.random.choice(X_a.shape[0],
                                   round(len(X_a) * X_perc) + 1 if round(len(X_a) * X_perc) > min_X else len(X_a),
                                   replace=False)]
        X_b = X[idx_iter][~cond]
        X_b = X_b[np.random.choice(X_b.shape[0],
                                   round(len(X_b) * X_perc) + 1 if round(len(X_b) * X_perc) > min_X else len(X_b),
                                   replace=False)]

        if len(X_a) == 0 or len(X_b) == 0:
            continue

        if np.any(is_categorical_feature) and not np.all(is_categorical_feature):  # mixed
            centroid_a = np.mean(X_a[:, ~is_categorical_feature], axis=0)
            centroid_b = np.mean(X_b[:, ~is_categorical_feature], axis=0)

            modoid_a = stats.mode(X_a[:, is_categorical_feature], axis=0, keepdims=True).mode[0]
            modoid_b = stats.mode(X_b[:, is_categorical_feature], axis=0, keepdims=True).mode[0]

            cm_a = np.zeros(n_features)
            cm_b = np.zeros(n_features)
            cm_a[~is_categorical_feature] = centroid_a
            cm_b[~is_categorical_feature] = centroid_b
            cm_a[is_categorical_feature] = modoid_a
            cm_b[is_categorical_feature] = modoid_b

            dist_a = cdist(X_a, cm_a.reshape(1, -1), metric=lambda u, v: _mixed_metric(con_indexes, cat_indexes, u, v))
            dist_b = cdist(X_b, cm_b.reshape(1, -1), metric=lambda u, v: _mixed_metric(con_indexes, cat_indexes, u, v))

        elif np.all(is_categorical_feature):  # all categorical
            modoid_a = stats.mode(X_a, axis=0, keepdims=True).mode[0]
            modoid_b = stats.mode(X_b, axis=0, keepdims=True).mode[0]

            dist_a = cdist(X_a, modoid_a.reshape(1, -1), metric=metric)
            dist_b = cdist(X_b, modoid_b.reshape(1, -1), metric=metric)

        else:  # all continuous
            centroid_a = np.mean(X_a, axis=0)
            centroid_b = np.mean(X_b, axis=0)

            dist_a = cdist(X_a, centroid_a.reshape(1, -1), metric=metric)
            dist_b = cdist(X_b, centroid_b.reshape(1, -1), metric=metric)

        mse_a = np.mean(dist_a)
        mse_b = np.mean(dist_b)
        mse_tot = (len(X_a) / (len(X_a) + len(X_b)) * mse_a + len(X_b) / (len(X_a)+ len(X_b)) * mse_b)

        if mse_tot < best_mse:
            best_feature = feature
            best_threshold = threshold
            best_mse = mse_tot

    return [best_mse, best_feature, best_threshold]
