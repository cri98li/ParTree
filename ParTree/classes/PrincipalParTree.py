import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import trange

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
            verbose=False
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

    def fit(self, X):
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot be hihger than X.shape[1]")
        super().fit(X)

    def _make_split(self, idx_iter):
        n_components_split = min(self.n_components, len(idx_iter))

        if len(self.con_indexes) == 0: #all caregorical
            transf = MCA(n_components=n_components_split, random_state=self.random_state)
        elif len(self.cat_indexes) == 0: #all continous
            transf = PCA(n_components=n_components_split, random_state=self.random_state)
        else: #mixed
            transf = FAMD(n_components=n_components_split, random_state=self.random_state)


        typed_X = pd.DataFrame(self.X[idx_iter])

        for index in self.cat_indexes:
            typed_X[index] = typed_X[index].apply(lambda x: f" {x}")

        typed_X.columns = typed_X.columns.astype(str)

        y_pca = transf.fit_transform(typed_X)

        clf_list = list()
        labels_list = list()
        bic_c_list = list()
        for i in trange(n_components_split, disable=not self.verbose):
            clf_i = DecisionTreeRegressor(
                max_depth=1,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
            clf_i.fit(self.X[idx_iter], y_pca[:, i])
            labels_i = clf_i.apply(self.X[idx_iter])
            bic_children_i = bic(self.X[idx_iter], (np.array(labels_i) - 1).tolist())

            clf_list.append(clf_i)
            labels_list.append(labels_i)
            bic_c_list.append(bic_children_i)

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
                olq_bic_children_i = bic(
                    self.X[idx_iter], (np.array(olq_labels_i) - 1).tolist()
                )

                clf_list.append(olq_clf_i)
                labels_list.append(olq_labels_i)
                bic_c_list.append(olq_bic_children_i)

        idx_min = np.argmin(bic_c_list)
        is_oblique = self.oblique_splits and idx_min > 0 and idx_min % 2 == 0
        labels = labels_list[idx_min]
        bic_children = bic_c_list[idx_min]
        clf = clf_list[idx_min]

        return clf, labels, bic_children, is_oblique
