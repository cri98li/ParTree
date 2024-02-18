import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import trange
from sklearn.metrics import r2_score
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
            mv=None,
            protected_attribute = None,
            def_type = None,
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
        self.mv = mv
        self.protected_attribute = protected_attribute
        self.def_type = def_type
        self.filename = filename

    def _calculate_similarity_matrix(self, points, threshold=9):
        points_arr = np.array(points)
        diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        adjacency_matrix = (dist_matrix < threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix

    def _write_to_file(self, content):
        #with open(filename, "a") as file:
        #    file.write(content + "\n")
        with open(self.filename, "a") as file:
            file.write(content + "\n")

    def _compute_penalty(self, points, labels):
        # users can choose the fairness definition
        # here they choose the individual fairness
        n = len(points)
        penalties = np.zeros(n)
        count_comp = 0
        # want to retrieve the row index of each point belonging to every cluster
        unique_labels = np.unique(labels)
        cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}


        if self.def_type == 'ind':
            similarity_matrix = self._calculate_similarity_matrix(points)
            for i in range(n):
                cluster_indices = np.where(labels == labels[i])[0]
            similar_count = np.sum(similarity_matrix[i, cluster_indices]) - similarity_matrix[i, i]
            penalty = similar_count/len(cluster_indices)
            penalties[i] = penalty

            return np.sum(penalties/len(penalties))

        # here users select the demographic parity definition
        elif self.def_type == 'dem':
            print("----- START NEW SPLIT -----")
            print("labels", labels)
            penalties = 0
            positive_prediction_rates = {}
            # we take each value of the protected attribute
            for cluster in set(labels):
                print("CLUSTER", cluster)
                X_filtered = self.X[cluster_indices[cluster]]
                for group in np.unique(X_filtered[:, self.protected_attribute]):
                    #print("group", group)
                    total_in_group = (X_filtered[:, self.protected_attribute]).shape[0]
                    positive_predictions_in_group = (
                                X_filtered[:, self.protected_attribute] == group).sum()
                    positive_prediction_rate = positive_predictions_in_group / total_in_group
                    positive_prediction_rates[group] = positive_prediction_rate
                keys = list(positive_prediction_rates.keys())
                print("keys", keys)
                for i, key1 in enumerate(keys):
                    if positive_prediction_rates[key1] == 1:
                        difference = 1
                        penalties += difference
                        count_comp += 1
                    for key2 in keys[i + 1:]:
                        if positive_prediction_rates[key1] == 1 and (key2 not in positive_prediction_rates or positive_prediction_rates[key2] == 1):
                            difference = 1
                        else:
                            difference = abs(positive_prediction_rates[key1] - positive_prediction_rates[key2])
                        penalties += difference
                        count_comp += 1

            print("return ", penalties/count_comp)
            return np.sum(penalties) / count_comp

        # third definition, group fairness in respect to proportion in the original dataset
        elif self.def_type == 'gro':
            print("----- START NEW SPLIT -----")

            group_cluster_counts = {}
            group_counts = {}
            total_cluster = {}
            penalties = 0

            # compute for each value of the protected attribute the number of points in each cluster
            for group in np.unique(self.X[:, self.protected_attribute]):
                print("unique values protected attribute", np.unique(self.X[:, self.protected_attribute]))
                group_counts[group] = (self.X[:, self.protected_attribute] == group).sum()
                for label in np.unique(labels):
                    total_cluster[label] = (labels == label).sum()
                    group_cluster_counts[(group, label)] = ((self.X[:, self.protected_attribute] == group) & (labels == label)).sum()

            total_count = self.X.shape[0]

            # Computation of the probability for each cluster and for each group
            for group in np.unique(self.X[:, self.protected_attribute]):
                total_in_group = group_counts[group]
                if total_in_group > 0:  # Prevent division by zero
                    tot_probability = total_in_group / total_count
                    # Proportion of group in the entire dataset
                    for label in np.unique(labels):
                        group_cluster_count = group_cluster_counts[(group, label)]
                        group_probability = group_cluster_count / total_cluster[label]  # Proportion of group in the cluster
                        diff = abs(tot_probability - group_probability)  # Compute difference in probabilities
                        penalties += diff
                        count_comp += 1

            return np.sum(penalties) / count_comp

        else:
            penalty = 0
            return penalty

    def fit(self, X):
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot be higher than X.shape[1]")
        super().fit(X)

    def _make_split(self, idx_iter):
        r2_children_i = 0
        n_components_split = min(self.n_components, len(idx_iter))

        if len(self.con_indexes) == 0:  # all categorical
            transf = MCA(n_components=n_components_split, random_state=self.random_state)
        elif len(self.cat_indexes) == 0:  # all continou
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

                best_split_score = -float('inf')
                best_split_value = None
                best_bic_score = None
                best_r2_score = None
                best_clf = None
                best_labels = None

                thresholds = sorted(np.unique(self.X[:, feature_index]))

                for idx_threshold in range(len(thresholds)-1):
                    value = thresholds[idx_threshold]
                    value_succ = thresholds[idx_threshold+1]

                    modified_X = np.zeros(self.X.shape)
                    modified_X[self.X[:, feature_index] <= value, feature_index] = value
                    modified_X[self.X[:, feature_index] > value, feature_index] = value_succ

                    # Train the Decision Tree on the modified feature
                    clf_i.fit(modified_X[idx_iter], transf.fit_transform(self.X[idx_iter]))
                    labels_i = clf_i.apply(modified_X[idx_iter])
                    #print("TO CHECK WITH THOSE labels_i", labels_i)
                    temp_score = clf_i.score(self.X[idx_iter], transf.fit_transform(self.X[idx_iter])) #da provare
                    bic_children_i = bic(self.X[idx_iter], (np.array(labels_i) - 1).tolist())

                    if self.def_type is not None:
                        penalty = self._compute_penalty(self.X[idx_iter], labels_i)
                    else:
                        penalty = 0
                    composite_score = temp_score - penalty

                    # Update the best split if this is better
                    if composite_score > best_split_score:
                        best_split_score = composite_score
                        best_split_value = value
                        best_bic_score = bic_children_i
                        best_r2_score = temp_score
                        best_clf = clf_i
                        best_labels = labels_i

                    # salvare valore split come value, penalty r2

                    self._write_to_file(f"\tFeature {feature_index}, Value {value}, Best Split Score {best_split_score}, Penalty {penalty}, R2 Score {best_r2_score}, BIC {bic_children_i}")

                # clusters
                if self.def_type == 'dem' or self.def_type == 'gro':
                    # Arrays to hold indexes
                    indexes_of_1 = []
                    indexes_of_2 = []

                    # Iterate through the list and append indexes accordingly
                    for index, value in enumerate(labels_i):
                        if value == 1:
                            indexes_of_1.append(index)
                        elif value == 2:
                            indexes_of_2.append(index)

                    protected_attribute_arr = self.X[:, self.protected_attribute]
                    cluster_1 = [int(protected_attribute_arr[index]) for index in indexes_of_1]
                    cluster_2 = [int(protected_attribute_arr[index]) for index in indexes_of_2]
                    similar_count = None
                elif self.def_type == 'ind':
                    points = self.X[idx_iter]
                    labels = clf_i.apply(modified_X[idx_iter])
                    n = len(points)
                    similarity_matrix = self._calculate_similarity_matrix(points)
                    for i in range(n):
                        cluster_indices = np.where(labels == labels[i])[0]
                    similar_count = np.sum(similarity_matrix[i, cluster_indices]) - similarity_matrix[i, i]
                    cluster_1 = None
                    cluster_2 = None
                else:
                    similar_count = None
                    cluster_1 = None
                    cluster_2 = None

                #print("LABELS LOG TO CHECK", labels_i)
                self._write_to_file(f"\nBest split for feature {feature_index}: Value {best_split_value}, Best Split Score {best_split_score}, R2 Score {best_r2_score}, Penalty {penalty}, BIC {best_bic_score}, \nLabels {labels_i}, Cluster 1 {cluster_1}, Cluster 2 {cluster_2}, [IND] similar count {similar_count}\n")

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

        idx_min = np.argmax(r2_c_list)
        is_oblique = self.oblique_splits and idx_min > 0 and idx_min % 2 == 0
        labels = labels_list[idx_min]
        bic_children = bic_c_list[idx_min]
        r2_children = r2_c_list[idx_min]
        clf = clf_list[idx_min]
        self._write_to_file(f"Overall Best Split: Feature Index {idx_min}, Bic Children {bic_children}, Is Oblique {is_oblique}, R2 Children {r2_children}\n")

        #print("labels", labels)
        return clf, labels, bic_children, is_oblique