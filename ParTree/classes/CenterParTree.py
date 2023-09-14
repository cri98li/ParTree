import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from ParTree.algorithms.bic_estimator import bic
from ParTree.algorithms.data_splitter import DecisionSplit
from ParTree.classes.ParTree import ParTree


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
            metric_con="cos",
            metric_cat="jaccard",
            n_jobs=1,
            verbose=False
    ):
        """
        For continuous features the algorithm uses "metric" distance, for categorical ones the Mode, and for mixed types
        it uses Seuclidean distance and Jaccard distance.

        :param metric_con:
            The distance metric to use. For more information see the cdist metric argument in the scipy documentation
        :param metric_cat:
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
        self.metric_cat = metric_cat
        self.metric_con = metric_con

    def _make_split(self, idx_iter):
        n_features = self.X.shape[1]

        best_feature = None
        best_threshold = None
        best_mse = np.inf

        results = []

        for n in tqdm(range(n_features), position=0, leave=False, disable=not self.verbose):
            for feature in tqdm(self.feature_values[n], position=1, leave=False, disable=not self.verbose):
                results.append(self.processPoolExecutor.submit(_make_split_innerloop,
                                                               self.X,
                                                               self.con_indexes,
                                                               self.cat_indexes,
                                                               self.metric_cat,
                                                               self.metric_con,
                                                               self.is_categorical_feature,
                                                               idx_iter,
                                                               n,
                                                               feature))

        for res in tqdm(results, disable=not self.verbose):
            best_returned_mse, best_returned_feature, best_returned_threshold = res.result()

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


def _mixed_metric(con_indexes, cat_indexes, u, v, metric_cat, metric_con):
    con_dist = np.inf
    if metric_con == "cos":
        con_dist = 1 - abs(cosine_similarity(u[con_indexes].reshape(1, -1), v[con_indexes].reshape(1, -1)))
    else:
        con_dist = cdist(u[con_indexes].reshape(1, -1), v[con_indexes].reshape(1, -1), metric=metric_con)

    cat_dist = 0
    if len(cat_indexes) != 0:
        cat_dist = cdist(u[cat_indexes].reshape(1, -1), v[cat_indexes].reshape(1, -1), metric=metric_cat)
    con_w = len(con_indexes) / len(u)
    cat_w = len(cat_indexes) / len(u)
    dist = con_w * con_dist + cat_w * cat_dist
    return dist


# dataset solo continui

def _make_split_innerloop(X, con_indexes, cat_indexes, metric_cat, metric_con, is_categorical_feature, idx_iter,
                          feature, threshold, X_perc=.2, min_X=1000):
    n_features = X.shape[1]

    cond = X[idx_iter, feature] == threshold if is_categorical_feature[feature] \
        else X[idx_iter, feature] <= threshold

    X_a = X[idx_iter][cond]
    X_a_sub = X_a[
        np.random.choice(X_a.shape[0], round(len(X_a) * X_perc) + 1 if round(len(X_a) * X_perc) > min_X else len(X_a),
                         replace=False)]
    X_b = X[idx_iter][~cond]
    X_b_sub = X_b[
        np.random.choice(X_b.shape[0], round(len(X_b) * X_perc) + 1 if round(len(X_b) * X_perc) > min_X else len(X_b),
                         replace=False)]

    if len(X_a) == 0 or len(X_b) == 0:
        return [np.inf, None, None]

    if np.any(is_categorical_feature) and not np.all(is_categorical_feature):
        # mixed (np.any(is_categorical_feature) and not np.all(is_categorical_feature))
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

        dist_a = cdist(X_a_sub, cm_a.reshape(1, -1), metric=lambda u, v: _mixed_metric(con_indexes, cat_indexes, u, v,
                                                                                       metric_con, metric_cat))
        dist_b = cdist(X_b_sub, cm_b.reshape(1, -1), metric=lambda u, v: _mixed_metric(con_indexes, cat_indexes, u, v,
                                                                                       metric_con, metric_cat))

    elif np.all(is_categorical_feature):  # all categorical
        modoid_a = stats.mode(X_a, axis=0, keepdims=True).mode[0]
        modoid_b = stats.mode(X_b, axis=0, keepdims=True).mode[0]

        dist_a = cdist(X_a_sub, modoid_a.reshape(1, -1), metric=metric_cat)
        dist_b = cdist(X_b_sub, modoid_b.reshape(1, -1), metric=metric_cat)

    else:  # all continuous
        centroid_a = np.mean(X_a, axis=0)
        centroid_b = np.mean(X_b, axis=0)

        dist_a = cdist(X_a_sub, centroid_a.reshape(1, -1), metric="euclidean")
        dist_b = cdist(X_b_sub, centroid_b.reshape(1, -1), metric="euclidean")

    mse_a = np.mean(dist_a)
    mse_b = np.mean(dist_b)
    # mse_tot = (len(X_a) / len(X[idx_iter]) * mse_a + len(X_b) / len(X[idx_iter]) * mse_b)
    mse_tot = (len(X_a) / (len(X_a) + len(X_b)) * mse_a + len(X_b) / (len(X_a) + len(X_b)) * mse_b)

    return [mse_tot, feature, threshold]
