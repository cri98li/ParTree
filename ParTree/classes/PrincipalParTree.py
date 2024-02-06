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
            def_type = None
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

    def _calculate_similarity_matrix(self, points, threshold=9):
        points_arr = np.array(points)
        diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        adjacency_matrix = (dist_matrix < threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix

    def _write_to_file(self, content):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"output_{timestamp}.txt"
        with open(filename, "a") as file:
            file.write(content + "\n")

    def _compute_penalty(self, points, labels):
        # users can choose the fairness definition
        # here they choose the individual fairness
        n = len(points)
        penalties = np.zeros(n)
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
            penalties = 0
            positive_prediction_rates = {}
            # we take each value of the protected attribute
            for cluster in set(labels):
                X_filtered = self.X[cluster_indices[cluster]]
                for group in np.unique(X_filtered[:, self.protected_attribute]):
                    total_in_group = (X_filtered[:, self.protected_attribute]).shape[0]
                    positive_predictions_in_group = (
                                X_filtered[:, self.protected_attribute] == group).sum()
                    positive_prediction_rate = positive_predictions_in_group / total_in_group
                    positive_prediction_rates[group] = positive_prediction_rate

                for group1 in positive_prediction_rates:
                    for group2 in positive_prediction_rates:
                        if group1 != group2:
                        # computation of the penalty as absolute difference between the prediction rates
                            difference = abs(positive_prediction_rates[group1] - positive_prediction_rates[group2])
                            penalties += difference
                            print("penalties", penalties)

            return np.sum(penalties/len(labels))
        # third definition, group fairness in respect to proportion in the original dataset
        elif self.def_type == 'gro':

            group_cluster_counts = {}
            group_counts = {}
            penalties = 0

            # compute for each value of the protected attribute the number of points in each cluster
            for group in np.unique(self.X[:, self.protected_attribute]):
                group_counts[group] = (self.X[:, self.protected_attribute] == group).sum()
                for label in np.unique(labels):
                    group_cluster_counts[(group, label)] = ((self.X[:, self.protected_attribute] == group) & (labels == label)).sum()

            total_count = self.X.shape[0]

            # Computation of the probability for each cluster and for each group
            for group in np.unique(self.X[:, self.protected_attribute]):
                total_in_group = group_counts[group]
                if total_in_group > 0:  # Prevent division by zero
                    tot_probability = total_in_group / total_count  # Proportion of group in the entire dataset
                    for label in np.unique(labels):
                        group_cluster_count = group_cluster_counts[(group, label)]
                        group_probability = group_cluster_count / total_in_group  # Proportion of group in the cluster
                        diff = abs(tot_probability - group_probability)  # Compute difference in probabilities
                        penalties += diff

            return np.sum(penalties) / len(labels)

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
        # bic_c_list = list()
        r2_c_list = list()

        for i in trange(n_components_split, disable=not self.verbose):
            clf_i = DecisionTreeRegressor(
                max_depth=1,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )

            output_info = ""

            for feature_index in range(self.X.shape[1]):
                best_split_score = float('inf')
                best_split_value = None
                best_r2_score = None

                for value in np.unique(self.X[:, feature_index]):
                    modified_X = np.zeros(self.X.shape)
                    modified_X[self.X[:, feature_index] <= value, feature_index] = value
                    modified_X[self.X[:, feature_index] > value, feature_index] = 2*(abs(value)+1)

                    # Train the Decision Tree on the modified feature
                    clf_i.fit(modified_X[idx_iter], transf.fit_transform(self.X[idx_iter]))
                    labels_i = clf_i.apply(modified_X[idx_iter])
                    temp_score = clf_i.score(self.X[idx_iter], transf.fit_transform(self.X[idx_iter])) #da provare
                    if self.def_type is not None:
                        penalty = self._compute_penalty(self.X[idx_iter], clf_i.apply(modified_X[idx_iter]))
                    else:
                        penalty = 0
                    composite_score = temp_score - (abs(temp_score)*penalty)

                    # Update the best split if this is better
                    if composite_score < best_split_score:
                        best_split_score = composite_score
                        best_split_value = value
                        best_r2_score = temp_score

                    # salvare valore split come value, penalty r2

                    split_info = f"Feature {feature_index}, Value {value}, Best Split Score {best_split_score}, Penalty {penalty}, R2 Score {best_r2_score}\n"
                    output_info += split_info

                self._write_to_file(f"Best split for feature {feature_index}: Value {best_split_value}, Best Split Score {best_split_score}, R2 Score {best_r2_score}, Penalty {penalty}")

                if self.def_type == 'dem':
                    #r2_children_i += abs(best_split_score) * penalty
                    r2_children_i = abs(best_split_score) * penalty
                else:
                    #r2_children_i += abs(best_split_score)
                    r2_children_i = abs(best_split_score)

                r2_c_list.append(r2_children_i)
                clf_list.append(clf_i)
                labels_list.append(labels_i)

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
                olq_r2_children_i = r2_score(y_pca[:, i], olq_predictions_i)

                clf_list.append(olq_clf_i)
                labels_list.append(olq_labels_i)
                r2_c_list.append(olq_r2_children_i)

        idx_min = np.argmin(r2_c_list)
        is_oblique = self.oblique_splits and idx_min > 0 and idx_min % 2 == 0
        labels = labels_list[idx_min]
        r2_children = r2_c_list[idx_min]
        clf = clf_list[idx_min]
        self._write_to_file(f"Overall Best Split: Feature Index {idx_min}, R2 Children {r2_children}, Is Oblique {is_oblique}")
        return clf, labels, r2_children, is_oblique
