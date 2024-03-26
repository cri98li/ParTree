import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import trange, tqdm
from sklearn.metrics import r2_score
from collections import Counter
from datetime import datetime

from ParTree.algorithms.bic_estimator import bic
from ParTree.algorithms.data_splitter import ObliqueHouseHolderSplit
from ParTree.algorithms.fairness_definitions import _fairness_gro, _fairness_dem, _fairness_ind
from ParTree.classes.ParTree import ParTree
from ParTree.light_famd import MCA, PCA, FAMD

from matplotlib import pyplot as plt
from sklearn import tree

from sklearn.base import clone


class PrincipalParTree(ParTree):
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
            n_components=1,
            oblique_splits=False,
            max_oblique_features=2,
            n_jobs=1,
            verbose=False,
            #mv=None,
            protected_attribute = None,
            #def_type = None,
            alfa_ind=None,
            alfa_dem=None,
            alfa_gro=None,
            filename = None
    ):
        """
        :param n_components:
            Number of components (must be less than the number of features in the dataset).

        :param oblique_splits:


        :param max_oblique_features:


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
        self.n_components = n_components
        self.oblique_splits = oblique_splits
        self.max_oblique_features = max_oblique_features
        #self.mv = mv
        self.protected_attribute = protected_attribute
        #self.def_type = def_type
        self.alfa_ind = alfa_ind
        self.alfa_dem = alfa_dem
        self.alfa_gro = alfa_gro
        self.filename = filename
        if not (0 <= alfa_ind <= 2) or not (0 <= alfa_gro <= 2) or not (0 <= alfa_dem <= 2):
            raise ValueError("Arguments must be within the range [0, 2]")

    def _write_to_file(self, content):
        #with open(filename, "a") as file:
        #    file.write(content + "\n")
        with open(self.filename, "a") as file:
            file.write(content + "\n")

    def fit(self, X):
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot be higher than X.shape[1]")
        super().fit(X)

    def _make_split(self, idx_iter):
        r2_children_i = 0
        n_components_split = min(self.n_components, len(idx_iter))
        print("self.con_indexes", self.con_indexes)
        print("self.cat_indexes", self.cat_indexes)

        if len(self.con_indexes) == 0:  # all categorical
            transf = MCA(n_components=n_components_split, random_state=self.random_state)
        elif len(self.cat_indexes) == 0:  # all continous
            transf = PCA(n_components=n_components_split, random_state=self.random_state)
        else:  # mixed
            transf = FAMD(n_components=n_components_split, random_state=self.random_state)

        typed_X = pd.DataFrame(self.X[idx_iter])

        for index in self.cat_indexes:
            typed_X[index] = typed_X[index].apply(lambda x: f" {x}")

        typed_X.columns = typed_X.columns.astype(str)

        y_pca = transf.fit_transform(typed_X)

        best_clf = None
        best_labels = None
        best_bic_score = None
        best_r2_score = -float("inf")
        best_is_oblique = None

        results = []

        for i in trange(n_components_split, disable=not self.verbose, position=0):
            for feature_index in trange(self.X.shape[1], position=1):
                #X, y_pca, feature_index, idx_iter, protected_attribute, alfa_ind, alfa_dem, alfa_gro,
                          #min_samples_leaf, min_samples_split, random_state, verbose
                results.append(self.processPoolExecutor.submit(_make_split_innerloop,
                                                               self.X[idx_iter],
                                                               y_pca[:, i],
                                                               feature_index,
                                                               self.protected_attribute,
                                                               self.alfa_ind,
                                                               self.alfa_dem,
                                                               self.alfa_gro,
                                                               self.min_samples_leaf,
                                                               self.min_samples_split,
                                                               self.random_state,
                                                               False
                                                               ))

        for idx_min, res in enumerate(tqdm(results, disable=not self.verbose)):
            best_returned_r2_score, best_returned_clf, best_returned_labels, best_returned_bic_score = res.result()

            if best_returned_r2_score > best_r2_score:
                best_clf = best_returned_clf
                best_labels = best_returned_labels
                best_bic_score = best_returned_bic_score
                best_r2_score = best_returned_r2_score
                best_is_oblique = self.oblique_splits and idx_min > 0 and idx_min % 2 == 0

        return best_clf, best_labels, best_bic_score, best_is_oblique


def _compute_penalty_old(points, labels, protected_attribute, alfa_ind, alfa_dem, alfa_gro):
    n = len(points)
    unique_labels = np.unique(labels)
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    penalty_ind = _fairness_ind(n, points, labels)
    penalty_dem = _fairness_dem(points, labels, cluster_indices, protected_attribute)
    penalty_gro = _fairness_gro(points, labels, protected_attribute)

    alfa = alfa_ind*penalty_ind + alfa_dem*penalty_dem + alfa_gro*penalty_gro

    return alfa

def _compute_penalty(points, labels, protected_attribute, alfa_ind, alfa_dem, alfa_gro):
    n = len(points)
    unique_labels = np.unique(labels)
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    penalty = .0

    if alfa_ind != 0:
        penalty += _fairness_ind(n, points, labels)*alfa_ind
    if alfa_dem != 0:
        penalty += _fairness_dem(points, labels, cluster_indices, protected_attribute)*alfa_dem
    if alfa_gro != 0:
        penalty += _fairness_gro(points, labels, protected_attribute)*alfa_gro

    return penalty

def _make_split_innerloop(X, y_pca, feature_index, protected_attribute, alfa_ind, alfa_dem, alfa_gro,
                          min_samples_leaf, min_samples_split, random_state, verbose):
    clf_i = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )

    # ("clf_i", clf_i)

    best_split_score = -float('inf')
    best_bic_score = None
    best_clf = None
    best_labels = None

    thresholds = sorted(np.unique(X[:, feature_index]))
    for idx_threshold in trange(len(thresholds) - 1, disable=not verbose):
        value = thresholds[idx_threshold]
        value_succ = thresholds[idx_threshold + 1]

        modified_X = np.zeros(X.shape)
        modified_X[X[:, feature_index] <= value, feature_index] = value
        modified_X[X[:, feature_index] > value, feature_index] = value_succ

        clf_i.fit(modified_X, y_pca)
        labels_i = clf_i.apply(modified_X)
        temp_score = clf_i.score(X, y_pca)
        bic_children_i = bic(X, (np.array(labels_i) - 1).tolist())

        alfa = _compute_penalty(X, labels_i, protected_attribute, alfa_ind, alfa_dem, alfa_gro)
        composite_score = temp_score - alfa  # *abs(temp_score-1)

        # Update the best split if this is better
        if composite_score > best_split_score:
            best_split_score = composite_score
            best_bic_score = bic_children_i
            best_clf = clf_i
            best_labels = labels_i

    # r2_c_list, clf_list, labels_list, bic_c_list
    return best_split_score, best_clf, best_labels, best_bic_score