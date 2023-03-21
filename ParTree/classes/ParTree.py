import heapq
import random
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import count
from typing import Union

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ParTree.algorithms.bic_estimator import bic
from ParTree.algorithms.data_preparation import prepare_data
from ParTree.algorithms.data_splitter import DecisionSplit
from ParTree.classes.ParTree_node import ParTree_node

global global_X


def init_pool(X):
    global global_X
    global_X = X


class ParTree(ABC):

    def __init__(
            self,
            max_depth: int = 3,
            max_nbr_clusters: int = 10,
            min_samples_leaf: int = 3,
            min_samples_split: int = 5,
            max_nbr_values: Union[int, float] = np.inf,
            max_nbr_values_cat: Union[int, float] = np.inf,
            bic_eps: float = 0.0,
            random_state: int = None,
            n_jobs: int = 1,
            verbose = False
    ):
        """
        Initialize the ParTree object.

        :param max_depth:
            Maximum depth of the tree describing the splits made by the algorithm. Consequently, with this parameter
            it is possible to limit the number of attribute tests in the antecedent.

        :param max_nbr_clusters:
            The maximum number of clusters to form as well as the number of centroids to generate.

        :param min_samples_leaf:
            The minimum number of samples required to be at a leaf node. A split point at any depth will only be
            considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.

        :param min_samples_split:
            The minimum number of samples required to split an internal node.

        :param max_nbr_values:
            Adjusts the maximum number of possible splits to be considered for continuous features. Given a feature,
            if there are more unique values than this hyperparameter, the values are binned


        :param max_nbr_values_cat:
            If the unique values of a feature do not exceed the value of this hyperparameter, it is treated by
            the algorithm as categorical

        :param bic_eps:
            Percentage of BIC parent discount.

        :param random_state:
            Parameter provided to the random.seed() function to make randomness deterministic.

        :param n_jobs:
            The number of jobs to run in parallel.

        :param verbose:

        """
        self.max_depth = max_depth
        self.max_nbr_clusters = max_nbr_clusters
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_nbr_values = max_nbr_values
        self.max_nbr_values_cat = max_nbr_values_cat
        self.bic_eps = bic_eps
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.processPoolExecutor = None

        random.seed(self.random_state)

        self.is_categorical_feature = None
        self.X = None
        self.labels_ = None
        self.clf_dict_ = None
        self.bic_ = None
        self.label_encoder_ = None
        self.cat_indexes = None
        self.feature_values = None
        self.con_indexes = None
        self.queue = list()

    def _make_leaf(self, node: ParTree_node):
        nbr_samples = len(node.idx)
        leaf_labels = np.array([node.label] * nbr_samples).astype(int)
        node_bic = bic(self.X[node.idx], [0] * nbr_samples)
        node.samples = nbr_samples
        node.support = nbr_samples / len(self.X)
        node.bic = node_bic
        node.is_leaf = True
        self.labels_[node.idx] = leaf_labels

    @abstractmethod
    def _make_split(self, idx_iter):
        pass

    def fit(self, X):
        tiebreaker = count()  # counter for the priority queue. Used in case of the same -len(idx)

        self.processPoolExecutor = ProcessPoolExecutor(self.n_jobs, initializer=init_pool, initargs=(X,))

        self.X = X
        n_features = X.shape[1]
        n_idx = X.shape[0]
        idx = np.arange(n_idx)

        self.labels_ = -1 * np.ones(n_idx).astype(int)

        cluster_id = 0
        root_node = ParTree_node(idx, cluster_id)

        heapq.heappush(self.queue, (-len(idx), (next(tiebreaker), idx, 0, root_node)))

        nbr_curr_clusters = 0

        self.feature_values, self.is_categorical_feature, X = prepare_data(X, self.max_nbr_values,
                                                                         self.max_nbr_values_cat)
        self.X = X

        self.con_indexes = np.array([i for i in range(n_features) if not self.is_categorical_feature[i]])
        self.cat_indexes = np.array([i for i in range(n_features) if self.is_categorical_feature[i]])

        while len(self.queue) > 0 and nbr_curr_clusters + len(self.queue) <= self.max_nbr_clusters:
            _, (_, idx_iter, node_depth, node) = heapq.heappop(self.queue)

            nbr_samples = len(idx_iter)

            if nbr_curr_clusters + len(self.queue) + 1 >= self.max_nbr_clusters \
                    or nbr_samples < self.min_samples_split \
                    or node_depth >= self.max_depth:
                self._make_leaf(node)
                nbr_curr_clusters += 1
                continue

            clf, labels, bic_children, is_oblique = self._make_split(idx_iter)

            if len(np.unique(labels)) == 1:
                self._make_leaf(node)
                nbr_curr_clusters += 1
                continue

            bic_parent = bic(self.X[idx_iter], [0] * nbr_samples)

            if bic_parent < bic_children - self.bic_eps * np.abs(bic_parent):
                self._make_leaf(node)
                nbr_curr_clusters += 1
                continue

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]

            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            cluster_id += 1
            node_l = ParTree_node(idx=idx_all_l, label=cluster_id)
            bic_l = bic(X[idx_iter[idx_l]], [0] * len(idx_l))

            cluster_id += 1
            node_r = ParTree_node(idx=idx_all_r, label=cluster_id)
            bic_r = bic(X[idx_iter[idx_r]], [0] * len(idx_r))

            node.clf = clf
            node.node_l = node_l
            node.node_r = node_r
            node.bic = bic_parent
            node.is_oblique = is_oblique

            heapq.heappush(self.queue, (-len(idx_all_l) + 0.00001 * bic_l, (next(tiebreaker), idx_all_l, node_depth + 1, node_l)))
            heapq.heappush(self.queue, (-len(idx_all_r) + 0.00001 * bic_r, (next(tiebreaker), idx_all_r, node_depth + 1, node_r)))

        self.clf_dict_ = root_node
        self.label_encoder_ = LabelEncoder()
        self.labels_ = self.label_encoder_.fit_transform(self.labels_)
        self.bic_ = bic(self.X, self.labels_)

    def predict(self, X):
        idx = np.arange(X.shape[0])
        labels = self._predict(X, idx, self.clf_dict_)
        labels = self.label_encoder_.transform(labels)
        return labels

    def _predict(self, X, idx, clf_dict):
        idx_iter = idx

        if clf_dict["clf"] is None:
            return np.array([clf_dict["label"]] * len(idx_iter))

        else:

            clf = clf_dict["clf"]
            labels = clf.apply(X[idx_iter])

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            labels_l = self._predict(X, idx_all_l, clf_dict["node_l"])
            labels_r = self._predict(X, idx_all_r, clf_dict["node_r"])

            labels[idx_l] = labels_l
            labels[idx_r] = labels_r

            return labels

    def get_axes2d(self, eps=1):
        idx = np.arange(self.X.shape[0])

        return self._get_axes2d(idx, self.clf_dict_, eps)

    def _get_axes2d(self, idx, clf_dict: ParTree_node, eps):
        idx_iter = idx

        axes2d = list()

        if clf_dict.clf is None:
            return []

        else:
            clf = clf_dict.clf
            labels = clf.apply(self.X[idx_iter])

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            x_min, x_max = self.X[idx_iter][:, 0].min(), self.X[idx_iter][:, 0].max()
            y_min, y_max = self.X[idx_iter][:, 1].min(), self.X[idx_iter][:, 1].max()

            if isinstance(clf, DecisionSplit):
                feat = clf.feature
                thr = clf.threshold

                if feat == 0:
                    axes = [[thr, thr], [y_min - eps, y_max + eps]]
                else:
                    axes = [[x_min - eps, x_max + eps], [thr, thr]]
            else:
                if not clf_dict.is_oblique:
                    feat = clf.tree_.feature[0]
                    thr = clf.tree_.threshold[0]

                    if feat == 0:
                        axes = [[thr, thr], [y_min - eps, y_max + eps]]
                    else:
                        axes = [[x_min - eps, x_max + eps], [thr, thr]]
                else:

                    def line_fun(x):
                        f = clf.oblq_clf.tree_.feature[0]
                        b = clf.oblq_clf.tree_.threshold[0]
                        m = (
                                clf.householder_matrix[:, f][0]
                                / clf.householder_matrix[:, f][1]
                        )
                        y = b - m * x - 1 + f
                        return y

                    axes = [
                        [x_min - eps, x_max + eps],
                        [line_fun(x_min - eps), line_fun(x_max + eps)],
                    ]

            axes2d.append(axes)

            axes2d += self._get_axes2d(idx_all_l, clf_dict.node_l, eps)
            axes2d += self._get_axes2d(idx_all_r, clf_dict.node_r, eps)

            return axes2d

    def get_rules(self):
        idx = np.arange(self.X.shape[0])
        return self._get_rules(idx, self.clf_dict_, 0)

    def _get_rules(self, idx_iter, clf_dict: ParTree_node, cur_depth):
        rules = list()

        if clf_dict.is_leaf:
            label = self.label_encoder_.transform([clf_dict.label])[0]
            leaf = (False, label, clf_dict.samples, clf_dict.support, cur_depth)

            rules.append(leaf)
            return rules

        else:
            clf = clf_dict.clf
            labels = clf.apply(self.X[idx_iter])

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            if isinstance(clf, DecisionSplit):
                feat = clf.feature
                thr = clf.threshold
                cat = clf.categorical
                rule = (True, [feat], [1.0], thr, cat, cur_depth)
            else:
                if not clf_dict.is_oblique:
                    feat = clf.tree_.feature[0]
                    thr = clf.tree_.threshold[0]
                    cat = feat in self.cat_indexes
                    rule = (True, [feat], [1.0], thr, cat, cur_depth)
                else:
                    pca_feat = clf.oblq_clf.tree_.feature[0]
                    thr = clf.oblq_clf.tree_.threshold[0]
                    feat_list = np.where(clf.u_weights != 0)[0].tolist()
                    coef = clf.householder_matrix[:, pca_feat][feat_list].tolist()
                    coef = StandardScaler().inverse_trasform(coef)
                    rule = (True, feat_list, coef, thr, False, cur_depth)

            rules.append(rule)
            rules += self._get_rules(idx_all_l, clf_dict.node_l, cur_depth + 1)
            rules += self._get_rules(idx_all_r, clf_dict.node_r, cur_depth + 1)
            return rules


def print_rules(rules, nbr_features, feature_names=None, precision=2, cat_precision=0):
    if feature_names is None:
        feature_names = ["X%s" % i for i in range(nbr_features)]

    s_rules = ""
    for rule in rules:
        is_rule = rule[0]
        depth = rule[-1]
        ident = "  " * depth
        if is_rule:
            _, feat_list, coef_list, thr, cat, _ = rule
            if len(feat_list) == 1:
                feat_s = "%s" % feature_names[feat_list[0]]
            else:
                feat_s = [
                    "%s %s"
                    % (np.round(coef_list[i], precision), feature_names[feat_list[i]])
                    for i in range(len(feat_list))
                ]
                feat_s = " + ".join(feat_s)
            if not cat:
                cond_s = "%s <= %s" % (feat_s, np.round(thr, precision))
            else:
                cond_s = "%s = %s" % (feat_s, np.round(thr, cat_precision))
                if cat_precision == 0:
                    cond_s = cond_s.replace(".0", "")
            s = "%s|-+ if %s:" % (ident, cond_s)
        else:
            _, label, samples, support, _ = rule
            support = np.round(support, precision)
            s = "%s|--> cluster: %s (%s, %s)" % (ident, label, samples, support)
        s_rules += "%s\n" % s

    return s_rules
