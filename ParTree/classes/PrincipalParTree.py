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
            # mv=None,
            protected_attribute=None,
            # def_type = None,
            alfa_ind=None,
            alfa_dem=None,
            alfa_gro=None,
            filename=None
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
        # self.mv = mv
        self.protected_attribute = protected_attribute
        # self.def_type = def_type
        self.alfa_ind = alfa_ind
        self.alfa_dem = alfa_dem
        self.alfa_gro = alfa_gro
        self.filename = filename

    def _calculate_similarity_matrix(self, points):
        points_arr = np.array(points)
        diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_matrix, np.nan)
        mean_distance = np.nanmean(dist_matrix)
        # print("dist matrix", diff)
        adjacency_matrix = (dist_matrix < mean_distance).astype(int)
        # print("adjacency_matrix", adjacency_matrix)
        # np.fill_diagonal(adjacency_matrix, 0)
        # print("adjacency_matrix", adjacency_matrix)
        return adjacency_matrix

    def _fairness_ind(self, alfa_ind, n, points, labels):
        # print("entrato")
        count_comp = 0
        penalties = 0
        similar_count_log = 0
        similarity_matrix = self._calculate_similarity_matrix(points)
        # print("similarity_matrix", similarity_matrix)
        for i in range(n):
            # print("i", i)
            cluster_indices = np.where(labels == labels[i])[0]
            # print("cluster_indices", cluster_indices)
            similar_count = np.sum(similarity_matrix[i, cluster_indices]) - similarity_matrix[i, i]
            similar_count_log += similar_count
            # print("similar_count", np.sum(similarity_matrix[i, cluster_indices]))
            # print("similarity_matrix[i, i]", similarity_matrix[i, i])
            # print("similarity_matrix[i, cluster_indices])", similarity_matrix[i, cluster_indices])
            penalty = similar_count / len(cluster_indices)
            # print("penalty", penalty)
            # print("len cluster indices", len(cluster_indices))
            count_comp += 1
            penalties += penalty
            # print("penalties", penalties)
            # print("penalty", penalty)
        # print("cluster indices", cluster_indices)
        # print("len cluster indices", len(cluster_indices))

        # print("count comp", count_comp)
        # print("penalties", np.sum(penalties))

        penalty = np.sum(penalties) / count_comp
        # print("alfa ind", alfa_ind)
        # print("penalty", penalty)
        # print("alfa_ind * penalty", alfa_ind * penalty)
        return alfa_ind * penalty

    def _fairness_dem(self, alfa_dem, points, labels, cluster_indices):
        # print("----- START NEW SPLIT -----")
        # print("labels", labels)
        penalties = 0
        count_comp = 0
        positive_prediction_rates = {}

        # we take each value of the protected attribute
        for cluster in set(labels):
            # print("set of labels", set(labels))
            # print("CLUSTER", cluster)
            # X_filtered = self.X[cluster_indices[cluster]]
            X_filtered = points[cluster_indices[cluster]]
            # print("cluster indices", cluster_indices[cluster])
            # for group in np.unique(X_filtered[:, self.protected_attribute]):
            for group in np.unique(points[:, self.protected_attribute]):
                # print("group", group)
                # print("distinct values over ALL the dataset", np.unique(points[:, self.protected_attribute]))
                total_in_group = (X_filtered[:, self.protected_attribute]).shape[0]
                positive_predictions_in_group = (
                        X_filtered[:, self.protected_attribute] == group).sum()
                # print("TEST --------------------",  X_filtered[:, self.protected_attribute] )
                # print("positive_predictions_in_group", positive_predictions_in_group)
                positive_prediction_rate = positive_predictions_in_group / total_in_group
                # print("positive_prediction_rate", positive_prediction_rate)
                positive_prediction_rates[group] = positive_prediction_rate
                # print("positive_prediction_rates", positive_prediction_rates)
            # print("positive_prediction_rates", positive_prediction_rates)

            # if len(positive_prediction_rates) < len(np.unique(points[:, self.protected_attribute])):
            #    for group in range(len(np.unique(points[:, self.protected_attribute])):
            #        if
            #        positive_prediction_rates[group] = 0

            keys = list(positive_prediction_rates.keys())
            # print("keys", keys)
            # for group1 in range(len(positive_prediction_rates)):

            # print("len(keys)", len(keys))
            # print("len(np.unique(points[:, self.protected_attribute]))",
            #      len(np.unique(points[:, self.protected_attribute])))

            # print("keys", keys)
            for i, key1 in enumerate(keys):
                # print("range positive_prediction_rates", range(len(positive_prediction_rates)))
                # print("positive_prediction_rates", positive_prediction_rates)
                # for group2 in range(group1 + 1, len(positive_prediction_rates)):
                # if positive_prediction_rates[key1] == 1:
                #    difference = 1
                #    penalties += difference
                #    count_comp += 1
                for key2 in keys[i + 1:]:
                    # print("entrato")
                    # if positive_prediction_rates[key1] == 1 and (key2 not in positive_prediction_rates or positive_prediction_rates[key2] == 1):
                    #    difference = 1
                    # print("key1", key1)
                    # print("positive_prediction_rates[key1]", positive_prediction_rates[key1])
                    # print("key2", key2)
                    # print("positive_prediction_rates[key1]", positive_prediction_rates[key2])
                    # print("difference", difference)
                    # else:
                    difference = abs(positive_prediction_rates[key1] - positive_prediction_rates[key2])
                    # print("key1", key1)
                    # print("positive_prediction_rates[key1]", positive_prediction_rates[key1])
                    # print("key2", key2)
                    # print("positive_prediction_rates[key1]", positive_prediction_rates[key2])
                    penalties += difference
                    count_comp += 1
                    # print("penalties", penalties)

            # for group1 in positive_prediction_rates:
            #    for group2 in positive_prediction_rates:
            #        if group1 != group2:
            #        # computation of the penalty as absolute difference between the prediction rates
            #            if positive_prediction_rates[group1] == 1 and positive_prediction_rates[group2] == 1:
            #                difference = 1
            #            else:
            #                difference = abs(positive_prediction_rates[group1] - positive_prediction_rates[group2])
            #            print("difference", difference)
            #            penalties += difference
            #            print("penalties", penalties)

        # print("return ", penalties/count_comp)
        penalty = np.sum(penalties) / count_comp
        # print("alfa_dem", alfa_dem)
        # print("penalty", penalty)
        # print("alfa_dem * penalty", alfa_dem * penalty)
        return alfa_dem * penalty

    def _fairness_gro(self, alfa_gro, points, labels):
        #print("----- START NEW SPLIT -----")

        group_cluster_counts = {}
        group_counts = {}
        total_cluster = {}
        count_comp = 0
        penalties = 0

        # compute for each value of the protected attribute the number of points in each cluster
        # for group in np.unique(self.X[:, self.protected_attribute]):
        for group in np.unique(points[:, self.protected_attribute]):
            # group_counts[group] = (self.X[:, self.protected_attribute] == group).sum()
            group_counts[group] = (points[:, self.protected_attribute] == group).sum()
            for label in np.unique(labels):
                # print("unique labels", np.unique(labels))
                total_cluster[label] = (labels == label).sum()
                # group_cluster_counts[(group, label)] = ((self.X[:, self.protected_attribute] == group) & (labels == label)).sum()
                group_cluster_counts[(group, label)] = (
                        (points[:, self.protected_attribute] == group) & (labels == label)).sum()
                # print("group", group)
                # print("label", label)
                # print("NUMBER OF DATAPOINTS OF GROUP & LABEL", group_cluster_counts[(group, label)])

        # total_count = self.X.shape[0]
        total_count = points.shape[0]
        # print("TOTALE DATAPOINTS (tutto dataset)", total_count)

        # Computation of the probability for each cluster and for each group
        # for group in np.unique(self.X[:, self.protected_attribute]):
        for group in np.unique(points[:, self.protected_attribute]):
            total_in_group = group_counts[group]
            # print("number of datapoints of the distinct value of protected attr", total_in_group)
            if total_in_group > 0:  # Prevent division by zero
                tot_probability = total_in_group / total_count
                # print("PROBABILITY DATASET", tot_probability)
                # Proportion of group in the entire dataset
                for label in np.unique(labels):
                    group_cluster_count = group_cluster_counts[(group, label)]
                    # print("[(group, label)]", [(group, label)])
                    # print("NUMBER OF DATAPOINTS OF GROUP & LABEL", group_cluster_counts[(group, label)])
                    group_probability = group_cluster_count / total_cluster[label]  # Proportion of group in the cluster
                    # print("tot probability", group_probability)
                    diff = abs(tot_probability - group_probability)  # Compute difference in probabilities
                    # print("probability dataset - group probability", diff)
                    penalties += diff
                    count_comp += 1

        # print("count_comp", count_comp)
        # return np.sum(penalties) / len(np.unique(self.X[:, self.protected_attribute]))
        penalty = np.sum(penalties) / len(np.unique(labels))
        # print("alfa_gro", alfa_gro)
        # print("penalty", penalty)
        # print("alfa_gro * penalty", alfa_gro * penalty)
        return alfa_gro * penalty

    def _write_to_file(self, content):
        # with open(filename, "a") as file:
        #    file.write(content + "\n")
        with open(self.filename, "a") as file:
            file.write(content + "\n")

    def _compute_penalty(self, points, labels):
        # users can choose the fairness definition
        # here they choose the individual fairness
        n = len(points)
        penalties = 0
        count_comp = 0
        # print("LABELS TO COMPARE TO THOSE IN LOGS", labels)
        # want to retrieve the row index of each point belonging to every cluster
        unique_labels = np.unique(labels)
        cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        # print("CLUSTER INDICES START PENALTY", cluster_indices)

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

                best_split_score = -float('inf')
                best_split_value = None
                best_bic_score = None
                best_r2_score = None
                best_clf = None
                best_labels = None

                ##### insert code to drop the protected attribute

                # print("X prima drop", self.X)
                # self.X = self.X.drop(self.X[:, self.protected_attribute], axis=1)
                # print("X dopo drop", X)

                thresholds = sorted(np.unique(self.X[:, feature_index]))
                modified_X = np.zeros(self.X.shape)
                for idx_threshold in range(len(thresholds) - 1):
                    value = thresholds[idx_threshold]
                    value_succ = thresholds[idx_threshold + 1]
                    #print("value, value_succ", value, value_succ)

                    modified_X = np.zeros(self.X.shape)
                    modified_X[self.X[:, feature_index] <= value, feature_index] = value
                    #print("modified_X[self.X[:, feature_index] <= value, feature_index]",
                          #modified_X[self.X[:, feature_index] <= value, feature_index])
                    modified_X[self.X[:, feature_index] > value, feature_index] = value_succ
                    #print("modified_X[self.X[:, feature_index] > value, feature_index]",
                          #modified_X[self.X[:, feature_index] > value, feature_index])

                    # modified_X = np.zeros(self.X.shape)
                    # modified_X[self.X[:, feature_index] <= value, feature_index] = value
                    # modified_X[self.X[:, feature_index] > value, feature_index] = value_succ

                    # print("modified_X", modified_X)

                    # Train the Decision Tree on the modified feature
                    clf_i.fit(modified_X[idx_iter], transf.fit_transform(self.X[idx_iter]))
                    labels_i = clf_i.apply(modified_X[idx_iter])
                    # print("TO CHECK WITH THOSE labels_i", labels_i)
                    # print("feature 14 original", self.X[idx_iter][:, 14])
                    # print("feature 14 modified", modified_X[idx_iter][:, 14])
                    # print("temp score 2 fit_transform same", transf.fit_transform(self.X[idx_iter]))
                    temp_score = clf_i.score(self.X[idx_iter], transf.fit_transform(self.X[idx_iter]))  # da provare
                    # print("r2 score", temp_score)
                    bic_children_i = bic(self.X[idx_iter], (np.array(labels_i) - 1).tolist())

                    # if self.def_type is not None:
                    alfa = self._compute_penalty(self.X[idx_iter], labels_i)
                    # else:
                    #    penalty = 0
                    composite_score = temp_score - alfa
                    # aggiungere alfa, più cresce più importanza in dominio
                    # aggiungere equalized odds
                    # calcolare tutte le definizioni di fairness e sottrarle tutte al composite score
                    # l'alfa diventa l'argomento, un alfa per definizione quindi tre alfa come argomento
                    # dominio di alfa tra 0 e 2
                    # nel composite score vogliamo un solo alfa, che però contiente tutti gli alfa
                    # parametri da passare: alfa generale come peso totale e alfa relativi al peso complessivo, gli alfa relativi sono sommati

                    # Update the best split if this is better
                    if composite_score > best_split_score:
                        best_split_score = composite_score
                        best_split_value = value
                        best_bic_score = bic_children_i
                        best_r2_score = temp_score
                        best_clf = clf_i
                        best_labels = labels_i

                    # salvare valore split come value, penalty r2

                    self._write_to_file(
                        f"\tFeature {feature_index}, Value {value}, Iteration Final Split Score {composite_score}, Penalty {alfa}, R2 Score {temp_score}, BIC {bic_children_i}")

                # clusters
                # if self.def_type == 'dem' or self.def_type == 'gro':
                # Arrays to hold indexes
                indexes_of_1 = []
                indexes_of_2 = []
                # print("labels", labels_i)
                # Iterate through the list and append indexes accordingly
                for index, value in enumerate(labels_i):
                    if value == 1:
                        indexes_of_1.append(index)
                    elif value == 2:
                        indexes_of_2.append(index)

                protected_attribute_arr = self.X[:, self.protected_attribute]
                # print("indexes_of_1",indexes_of_1)
                cluster_1 = [int(protected_attribute_arr[index]) for index in indexes_of_1]
                cluster_1 = Counter(cluster_1)
                cluster_2 = [int(protected_attribute_arr[index]) for index in indexes_of_2]
                cluster_2 = Counter(cluster_2)

                similar_count = None
                similar_count_log = None
                # elif self.def_type == 'ind':
                points = self.X[idx_iter]
                labels = clf_i.apply(modified_X[idx_iter])
                n = len(points)
                similar_count_log = 0
                similarity_matrix = self._calculate_similarity_matrix(points)
                for i in range(n):
                    cluster_indices = np.where(labels == labels[i])[0]
                    similar_count = np.sum(similarity_matrix[i, cluster_indices]) - similarity_matrix[i, i]
                    similar_count_log += similar_count
                # cluster_1 = None
                # cluster_2 = None
                # else:
                # similar_count_log = None
                # cluster_1 = None
                # cluster_2 = None

                # print("LABELS LOG TO CHECK", labels_i)
                self._write_to_file(
                    f"\nBest split for feature {feature_index}: Value {best_split_value}, Best Split Score {best_split_score}, R2 Score {best_r2_score}, Penalty {alfa}, BIC {best_bic_score}, Cluster 1 {cluster_1}, Cluster 2 {cluster_2}, [IND] similar count {similar_count_log}\n")

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
        self._write_to_file(
            f"Overall Best Split: Feature Index {idx_min}, Bic Children {bic_children}, Is Oblique {is_oblique}, R2 Children {r2_children}\n")

        # print("labels", labels)
        return clf, labels, bic_children, is_oblique