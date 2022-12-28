import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor

from ParTree.algorithms.bic_estimator import bic
from ParTree.algorithms.data_splitter import ObliqueHouseHolderSplit
from ParTree.classes import ParTree

from tqdm.auto import tqdm, trange


class PrincipalParTree(ParTree):
    def __init__(
        self,
        max_depth=3,
        max_nbr_clusters=10,
        min_samples_leaf=3,
        min_samples_split=5,
        max_nbr_values=np.inf,
        max_nbr_values_cat=np.inf,
        random_state=None,
        bic_eps=0.0,
        n_components=1,
        oblique_splits=False,
        max_oblique_features=2,
    ):
        super().__init__(
            max_depth,
            max_nbr_clusters,
            min_samples_leaf,
            min_samples_split,
            max_nbr_values,
            max_nbr_values_cat,
            bic_eps,
            random_state,
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
        pca = PCA(n_components=n_components_split)
        y_pca = pca.fit_transform(self.X[idx_iter])

        clf_list = list()
        labels_list = list()
        bic_c_list = list()
        for i in trange(n_components_split):
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
                    pca=pca,
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