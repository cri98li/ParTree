import itertools
import os
import time

import numpy as np
import pandas as pd
import ParTree.algorithms.measures_utils as measures

import psutil
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm


class KMeansTree():

    def __init__(self,
                 n_clusters=8,
                 labels_as_tree_leaves=True,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 tol=0.0001,
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 algorithm='auto',
                 criterion='gini',
                 splitter='best',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 ccp_alpha=0.0):
        self.kmeans = KMeans(n_clusters=n_clusters,
                             init=init,
                             n_init=n_init,
                             max_iter=max_iter,
                             tol=tol,
                             verbose=verbose,
                             random_state=random_state,
                             copy_x=copy_x,
                             algorithm=algorithm)

        max_depth = np.round(np.log2(n_clusters)).astype(int)
        self.dt = DecisionTreeClassifier(criterion=criterion,
                                         splitter=splitter,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                                         max_features=max_features,
                                         random_state=random_state,
                                         max_leaf_nodes=max_leaf_nodes,
                                         min_impurity_decrease=min_impurity_decrease,
                                         class_weight=class_weight,
                                         ccp_alpha=ccp_alpha)
        self.labels_ = None
        self.labels_as_tree_leaves = labels_as_tree_leaves

    def fit(self, X):
        self.kmeans.fit(X)
        self.dt.fit(X, self.kmeans.labels_)
        if self.labels_as_tree_leaves:
            self.labels_ = self.dt.apply(X)
        else:
            self.labels_ = self.dt.predict(X)

    def predict(self, X):
        if self.labels_as_tree_leaves:
            self.labels_ = self.dt.apply(X)
        else:
            self.labels_ = self.dt.predict(X)


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import silhouette_score, v_measure_score, normalized_mutual_info_score

from sklearn.cluster import KMeans


def run(datasets: str, destination_folder: str):
    runs = [
        # ("CenterParTree2", run_CenterParTree),
        # ("ImpurityParTree2", run_ImpurityParTree),
        ("kmeans+DT", run_kmeansPartree)
    ]

    datasets_bar = tqdm(datasets, position=0, leave=False)
    for dataset in datasets_bar:
        dataset_name = dataset.split('\\')[-1].split('/')[-1]
        datasets_bar.set_description(f"Dataset name: {dataset_name}")

        f_bar = tqdm(runs, position=1, leave=False)
        for (name, f) in f_bar:
            f_bar.set_description(f"Algorithm: {name}")
            f(dataset, destination_folder)


def run_kmeansPartree(dataset: str, res_folder):
    """
     init='k-means++',
     n_init=10,
     max_iter=300,
     tol=0.0001,
     verbose=0,
     random_state=None,
     copy_x=True,
     algorithm='auto',
     criterion='gini',
     splitter='best',
     min_samples_split=2,
     min_samples_leaf=1,
     min_weight_fraction_leaf=0.0,
     max_features=None,
     max_leaf_nodes=None,
     min_impurity_decrease=0.0,
     class_weight=None,
     ccp_alpha=0.0

    """

    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["n_clusters", "labels_as_tree_leaves", "init", "n_init", "max_iter", "max_iter", "tol",
                        "verbose", "random_state", "copy", "algorithm", "criterion", "splitter", "min_samples_split",
                        "min_samples_leaf", "min_weight_fraction_leaf", "max_features", "max_leaf_nodes",
                        "min_impurity_decrease", "class_weight", "ccp_alpha"]

    parameters = [
        [len(np.unique(y))] if has_y else range(2, 12 + 1, 2),  # n_clusters
        [True],  # labels_as_tree_leaves
        ["k-means++"],  # init
        [11],  # n_init
        [100, 300, 500],  # max_iter
        [.0001],  # tol
        [False],  # verbose
        [42],  # random_state
        [True],  # copy,
        ["lloyd"],  # algorithm
        ["gini"],  # criterion
        ["best"],  # splitter
        [3, 30],  # min_samples_split
        [5, 50],  # min_samples_leaf
        [.0],  # min_weight_fraction_leaf
        [None],  # max_features
        [None],  # max_leaf_nodes
        [.0],  # min_impurity_decrease
        [None],  # class_weight
        [.0],  # ccp_alpha
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "kmeans+dt-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            cpt = KMeansTree(els[0], els[1], els[2], els[3], els[4], els[5], els[6], els[7], els[8], els[9], els[10],
                             els[11], els[12], els[13], els[14], els[15], els[16], els[17], els[18], els[19])

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)

            start = time.time()
            cpt.fit(X)
            stop = time.time()

            row = list(els) + [stop - start] + measures.get_metrics_uns(X, cpt.labels_)
            if has_y:
                row += measures.get_metrics_s(cpt.labels_, y)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'}")
            raise e


if __name__ == '__main__':
    run(["datasets/real/adult.zip"])


def get_name():
    return "kmeans+dt"


def get_version():
    return "-1"
