import itertools
import os.path
import time
from time import sleep

import numpy as np
import pandas as pd
import pkg_resources
import psutil as psutil
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from ParTree.classes.CenterParTree import CenterParTree
import ParTree.algorithms.measures_utils as measures


def run(datasets: list, destination_folder: str):
    runs = [run_CenterParTree]

    for dataset in tqdm(datasets, position=0):
        for f in tqdm(runs, position=1):
            f(dataset)


def run_CenterParTree(dataset: str):

    is_syntetic = "syntetic" in dataset

    df = pd.read_csv(dataset, header=None, index_col=None)
    y = None
    if is_syntetic:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["max_depth", "max_nbr_clusters", "min_samples_leaf", "min_samples_split", "max_nbr_values",
                        "max_nbr_values_cat", "bic_eps", "random_state", "metric"]

    parameters = [
        len(y) if is_syntetic else range(2, 10, 3),  # max_depth
        range(2, 13, 3),  # max_nbr_clusters
        [3],  # range(1, 100, 30),  # min_samples_leaf
        [5],  # range(2, 100, 30),  # min_samples_split
        range(10, 200 + 1, 50),  # max_nbr_values
        range(5, 21, 5),  # max_nbr_values_cat
        np.arange(.0, .3, .1),  # bic_eps
        [42],  # random_state
        ["euclidean"]  # metric
    ]

    for els in tqdm(list(itertools.product(*parameters)), position=3):
        colNames = hyperparams_name+["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
        if is_syntetic:
            colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                         "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

        filename = "_".join([str(x) for x in els])+".csv"
        if os.path.exists(filename):
            continue

        cpt = CenterParTree(els[0], els[1], els[2], els[3], els[4], els[5], els[6], els[7], els[8],
                            psutil.cpu_count(logical=False))

        ct = ColumnTransformer([
            ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
            ("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
            remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

        X = ct.fit_transform(df)

        start = time.time()
        cpt.fit(X)
        stop = time.time()


        row = list(els) + [stop-start] + measures.get_metrics_uns(X, cpt.labels_)
        if is_syntetic:
            row += measures.get_metrics_s(cpt.labels_, y)

        pd.DataFrame([row], columns=colNames).to_csv(filename)




def get_name():
    return "ParTree"


def get_version():
    return pkg_resources.get_distribution("ParTree").version


if __name__ == '__main__':
    run(["datasets/real/adult.zip"])
