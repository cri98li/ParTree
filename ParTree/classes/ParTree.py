import heapq
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ParTree.algorithms.bic_estimator import bic
from ParTree.algorithms.data_splitter import DecisionSplit
from ParTree.classes.ParTree_node import ParTree_node


class ParTree(ABC):
    def __init__(
        self,
        max_depth=3,
        max_nbr_clusters=10,
        min_samples_leaf=3,
        min_samples_split=5,
        max_nbr_values=np.inf,
        max_nbr_values_cat=np.inf,
        bic_eps=0.0,
        random_state=None,
        n_jobs=1
    ):
        self.max_depth = max_depth
        self.max_nbr_clusters = max_nbr_clusters
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_nbr_values = max_nbr_values
        self.max_nbr_values_cat = max_nbr_values_cat
        self.bic_eps = bic_eps
        self.random_state = random_state
        self.processPoolExecutor = ProcessPoolExecutor(n_jobs)

        random.seed(self.random_state)

        self.X = None
        self.labels_ = None
        self.clf_dict_ = None
        self.bic_ = None
        self.label_encoder_ = None

    def _build_leaf(self, node: ParTree_node):
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
        self.X = X
        idx = np.arange(self.X.shape[0])

        self.labels_ = -1 * np.ones(self.X.shape[0]).astype(int)

        self.queue = list()
        iter_count = 0

        cluster_id = 0
        root_node = ParTree_node(idx, cluster_id)

        heapq.heappush(self.queue, (-len(idx), (idx, 0, root_node)))

        nbr_curr_clusters = 0

        self.feature_values = dict()
        n_features = self.X.shape[1]
        self.is_categorical_feature = [False] * n_features
        for feature in range(n_features):
            values = np.unique(self.X[:, feature])
            if len(values) > self.max_nbr_values:
                _, vals = np.histogram(values, bins=self.max_nbr_values)
                values = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            self.feature_values[feature] = values

            if len(values) <= self.max_nbr_values_cat:
                self.is_categorical_feature[feature] = True

        self.con_indexes = np.array(
            [i for i in range(n_features) if not self.is_categorical_feature[i]]
        )
        self.cat_indexes = np.array(
            [i for i in range(n_features) if self.is_categorical_feature[i]]
        )

        while (
            len(self.queue) > 0
            and nbr_curr_clusters + len(self.queue) <= self.max_nbr_clusters
        ):

            iter_count += 1
            _, vals = heapq.heappop(self.queue)
            idx, node_depth, node = vals

            idx_iter = idx
            nbr_samples = len(idx_iter)

            if nbr_curr_clusters + len(self.queue) + 1 >= self.max_nbr_clusters\
                    or nbr_samples < self.min_samples_split\
                    or node_depth >= self.max_depth:
                self._build_leaf(node)
                nbr_curr_clusters += 1
                continue

            clf, labels, bic_children, is_oblique = self._make_split(idx_iter)

            if len(np.unique(labels)) == 1:
                self._build_leaf(node)
                nbr_curr_clusters += 1
                continue

            bic_parent = bic(self.X[idx_iter], [0] * nbr_samples)

            if bic_parent < bic_children - self.bic_eps * np.abs(bic_parent):
                self._build_leaf(node)
                nbr_curr_clusters += 1
                continue

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]

            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            cluster_id += 1
            node_l = ParTree_node(idx_all_l, cluster_id)
            bic_l = bic(self.X[idx_iter[idx_l]], [0] * len(idx_l))

            cluster_id += 1
            node_r = ParTree_node(idx_all_r, cluster_id)
            bic_r = bic(self.X[idx_iter[idx_r]], [0] * len(idx_r))

            node.clf = clf
            node.node_l = node_l
            node.node_r = node_r
            node.bic = bic_parent
            node.is_oblique = is_oblique

            heapq.heappush(
                self.queue,
                (
                    -len(idx_all_l) + 0.00001 * bic_l,
                    (idx_all_l, node_depth + 1, node_l),
                ),
            )

            heapq.heappush(
                self.queue,
                (
                    -len(idx_all_r) + 0.00001 * bic_r,
                    (idx_all_r, node_depth + 1, node_r),
                ),
            )

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
        axes2d = list()
        self._get_axes2d(idx, self.clf_dict_, axes2d, eps)

        return axes2d

    def _get_axes2d(self, idx, clf_dict, axes2d, eps):
        idx_iter = idx

        if clf_dict["clf"] is None:
            return

        else:
            clf = clf_dict["clf"]
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
                if not clf_dict["is_oblique"]:
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

            self._get_axes2d(idx_all_l, clf_dict["node_l"], axes2d, eps)
            self._get_axes2d(idx_all_r, clf_dict["node_r"], axes2d, eps)

    def get_rules(self):
        idx = np.arange(self.X.shape[0])
        rules = list()
        self._get_rules(idx, self.clf_dict_, rules, 0)
        return rules

    def _get_rules(self, idx, clf_dict, rules, cur_depth):
        idx_iter = idx

        if clf_dict["is_leaf"]:
            label = self.label_encoder_.transform([clf_dict["label"]])[0]
            leaf = (False, label, clf_dict["samples"], clf_dict["support"], cur_depth)

            rules.append(leaf)
            return

        else:
            clf = clf_dict["clf"]
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
                if not clf_dict["is_oblique"]:
                    feat = clf.tree_.feature[0]
                    thr = clf.tree_.threshold[0]
                    rule = (True, [feat], [1.0], thr, False, cur_depth)
                else:
                    pca_feat = clf.oblq_clf.tree_.feature[0]
                    thr = clf.oblq_clf.tree_.threshold[0]
                    feat_list = np.where(clf.u_weights != 0)[0].tolist()
                    coef = clf.householder_matrix[:, pca_feat][feat_list].tolist()
                    coef = StandardScaler().inverse_trasform(coef)
                    rule = (True, feat_list, coef, thr, False, cur_depth)

            rules.append(rule)
            self._get_rules(idx_all_l, clf_dict["node_l"], rules, cur_depth + 1)
            self._get_rules(idx_all_r, clf_dict["node_r"], rules, cur_depth + 1)
            return


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
