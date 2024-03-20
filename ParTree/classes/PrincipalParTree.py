import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import trange
from sklearn.metrics import r2_score
from collections import Counter
from datetime import datetime

from ParTree.algorithms.bic_estimator import bic
from ParTree.algorithms.data_splitter import ObliqueHouseHolderSplit
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

    def _calculate_similarity_matrix(self, points):
        points_arr = np.array(points)
        diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_matrix, np.nan)
        mean_distance = np.nanmean(dist_matrix)
        adjacency_matrix = (dist_matrix < mean_distance).astype(int)
        return adjacency_matrix

    def _fairness_ind(self, alfa_ind, n, points, labels):
        count_comp = 0
        penalties = 0
        similar_count_log = 0
        similarity_matrix = self._calculate_similarity_matrix(points)
        for i in range(n):
            cluster_indices = np.where(labels == labels[i])[0]
            similar_count = np.sum(similarity_matrix[i, cluster_indices]) - similarity_matrix[i, i]
            similar_count_log += similar_count
            penalty = similar_count / len(cluster_indices)
            count_comp += 1
            penalties += penalty

        penalty = np.sum(penalties) / count_comp
        return alfa_ind * penalty

    def _fairness_dem(self, alfa_dem, points, labels, cluster_indices):
        penalties = 0
        count_comp = 0
        positive_prediction_rates = {}

        # we take each value of the protected attribute
        for cluster in set(labels):
            X_filtered = points[cluster_indices[cluster]]
            for group in np.unique(points[:, self.protected_attribute]):
                total_in_group = (X_filtered[:, self.protected_attribute]).shape[0]
                positive_predictions_in_group = (
                        X_filtered[:, self.protected_attribute] == group).sum()
                positive_prediction_rate = positive_predictions_in_group / total_in_group
                positive_prediction_rates[group] = positive_prediction_rate

            keys = list(positive_prediction_rates.keys())

            for i, key1 in enumerate(keys):
                for key2 in keys[i + 1:]:
                    # else:
                    difference = abs(positive_prediction_rates[key1] - positive_prediction_rates[key2])
                    penalties += difference
                    count_comp += 1

        penalty = np.sum(penalties) / count_comp
        return alfa_dem * penalty

    def _fairness_gro(self, alfa_gro, points, labels):

        group_cluster_counts = {}
        group_counts = {}
        total_cluster = {}
        count_comp = 0
        penalties = 0

        for group in np.unique(points[:, self.protected_attribute]):
            # group_counts[group] = (self.X[:, self.protected_attribute] == group).sum()
            group_counts[group] = (points[:, self.protected_attribute] == group).sum()
            for label in np.unique(labels):
                total_cluster[label] = (labels == label).sum()
                group_cluster_counts[(group, label)] = (
                        (points[:, self.protected_attribute] == group) & (labels == label)).sum()

        total_count = points.shape[0]

        for group in np.unique(points[:, self.protected_attribute]):
            total_in_group = group_counts[group]
            if total_in_group > 0:  # Prevent division by zero
                tot_probability = total_in_group / total_count
                for label in np.unique(labels):
                    group_cluster_count = group_cluster_counts[(group, label)]
                    group_probability = group_cluster_count / total_cluster[label]
                    diff = abs(tot_probability - group_probability)
                    penalties += diff
                    count_comp += 1

        penalty = np.sum(penalties) / len(np.unique(labels))
        return alfa_gro * penalty

    def _write_to_file(self, content):
        #with open(filename, "a") as file:
        #    file.write(content + "\n")
        with open(self.filename, "a") as file:
            file.write(content + "\n")

    def _compute_penalty(self, points, labels):
        n = len(points)
        penalties = 0
        count_comp = 0
        unique_labels = np.unique(labels)
        cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

        penalty_ind = self._fairness_ind(self.alfa_ind, n, points, labels)
        penalty_dem = self._fairness_dem(self.alfa_dem, points, labels, cluster_indices)
        penalty_gro = self._fairness_gro(self.alfa_gro, points, labels)

        alfa = penalty_ind + penalty_dem + penalty_gro

        return alfa

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

        clf_list = list()
        labels_list = list()
        bic_c_list = list()
        r2_c_list = list()

        for i in trange(n_components_split, disable=not self.verbose):
            for feature_index in range(self.X.shape[1]):
                print("******************feature**************+****", feature_index)
                clf_i = DecisionTreeRegressor(
                    max_depth=3,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )

                #("clf_i", clf_i)

                best_split_score = -float('inf')
                best_split_value = None
                best_bic_score = None
                best_r2_score = None
                best_clf = None
                best_labels = None

                thresholds = sorted(np.unique(self.X[:, feature_index]))
                modified_X = np.zeros(self.X.shape)
                print("range(len(thresholds) - 1)", range(len(thresholds) - 1))
                for idx_threshold in range(len(thresholds) - 1):
                    value = thresholds[idx_threshold]
                    value_succ = thresholds[idx_threshold + 1]
                    #print("value, value_succ", value, value_succ)

                    modified_X = np.zeros(self.X.shape)
                    modified_X[self.X[:, feature_index] <= value, feature_index] = value
                    modified_X[self.X[:, feature_index] > value, feature_index] = value_succ
                    #print("clf_i.fit(modified_X[idx_iter], y_pca)", clf_i.fit(modified_X[idx_iter], y_pca))
                    #print("y_pca", y_pca)
                    #print("modified_X[idx_iter]", modified_X[idx_iter])
                    clf_i.fit(modified_X[idx_iter], y_pca)
                    labels_i = clf_i.apply(modified_X[idx_iter])
                    temp_score = clf_i.score(self.X[idx_iter], y_pca)
                    bic_children_i = bic(self.X[idx_iter], (np.array(labels_i) - 1).tolist())

                    alfa = self._compute_penalty(self.X[idx_iter], labels_i)
                    #else:
                    #    penalty = 0
                    composite_score = temp_score - alfa

                    # Update the best split if this is better
                    if composite_score > best_split_score:
                        best_split_score = composite_score
                        best_split_value = value
                        best_bic_score = bic_children_i
                        best_r2_score = temp_score
                        best_clf = clf_i
                        best_labels = labels_i

                    #self._write_to_file(f"\tFeature {feature_index}, Value {value}, Iteration Final Split Score {composite_score}, Penalty {alfa}, R2 Score {temp_score}, BIC {bic_children_i}")

                # clusters
                #if self.def_type == 'dem' or self.def_type == 'gro':
                    # Arrays to hold indexes
               # indexes_of_1 = []
               # indexes_of_2 = []
                    #print("labels", labels_i)
                    # Iterate through the list and append indexes accordingly
               # for index, value in enumerate(labels_i):
               #     if value == 1:
               #         indexes_of_1.append(index)
               #     elif value == 2:
               #         indexes_of_2.append(index)

               # protected_attribute_arr = self.X[:, self.protected_attribute]
                    #print("indexes_of_1",indexes_of_1)
               # cluster_1 = [int(protected_attribute_arr[index]) for index in indexes_of_1]
               # cluster_1 = Counter(cluster_1)
               # cluster_2 = [int(protected_attribute_arr[index]) for index in indexes_of_2]
               # cluster_2 = Counter(cluster_2)

               # similar_count = None
               # similar_count_log = None
               # elif self.def_type == 'ind':
               # points = self.X[idx_iter]
               # labels = clf_i.apply(modified_X[idx_iter])
               # #print("LABELS", labels)
               # n = len(points)
               # similar_count_log = 0
               # similarity_matrix = self._calculate_similarity_matrix(points)
               # for i in range(n):
               #     cluster_indices = np.where(labels == labels[i])[0]
               #     similar_count = np.sum(similarity_matrix[i, cluster_indices]) - similarity_matrix[i, i]
               #     similar_count_log += similar_count

                #self._write_to_file(f"\nBest split for feature {feature_index}: Value {best_split_value}, Best Split Score {best_split_score}, R2 Score {best_r2_score}, Penalty {alfa}, BIC {best_bic_score}, Cluster 1 {cluster_1}, Cluster 2 {cluster_2}, [IND] similar count {similar_count_log}\n")

                r2_c_list.append(best_split_score)
                clf_list.append(best_clf)
                labels_list.append(best_labels)
                bic_c_list.append(best_bic_score)

            if self.oblique_splits and i > 0:
                olq_clf_i = ObliqueHouseHolderSplit(
                    pca=transf,
                    max_oblique_features=self.max_oblique_features,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )

                olq_clf_i.fit(self.X[idx_iter], y_pca[:, i])
                olq_labels_i = olq_clf_i.apply(self.X[idx_iter])
                olq_r2_children_i = r2_score(self.X[idx_iter], (np.array(labels_i) - 1).tolist())

                clf_list.append(olq_clf_i)
                labels_list.append(olq_labels_i)
                r2_c_list.append(olq_r2_children_i)

        #print("Il numero due di r2_c_list", r2_c_list)
        #print("bic_c_list", bic_c_list)
        idx_min = np.argmax(r2_c_list)
        is_oblique = self.oblique_splits and idx_min > 0 and idx_min % 2 == 0
        #print("labels list before label", labels_list)
        #print("idxmin", idx_min)
        labels = labels_list[idx_min]
        bic_children = bic_c_list[idx_min]
        r2_children = r2_c_list[idx_min]
        clf = clf_list[idx_min]
        #self._write_to_file(f"Overall Best Split: Feature Index {idx_min}, Bic Children {bic_children}, Is Oblique {is_oblique}, R2 Children {r2_children}\n")
        #print("clf, labels, bic_children, is_oblique", clf, labels, bic_children, is_oblique)
        return clf, labels, bic_children, is_oblique