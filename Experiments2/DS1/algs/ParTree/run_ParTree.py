import itertools
import os.path
import time
import warnings
from glob import glob
from time import sleep

import numpy as np
import pandas as pd
import pkg_resources
import psutil as psutil
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from tqdm.auto import tqdm

from ParTree.classes.CenterParTree import CenterParTree
from ParTree.classes.ImpurityParTree import ImpurityParTree
from ParTree.classes.PrincipalParTree import PrincipalParTree


def run(datasets: list, destination_folder: str):
    runs = [
        ("CenterParTree", run_CenterParTree),
        ("PrincipalParTree", run_PrincipalParTree),
        ("ImpurityParTree", run_ImpurityParTree),
    ]

    datasets_bar = tqdm(datasets, position=0, leave=False, dynamic_ncols=True)
    for dataset in datasets_bar:
        dataset_name = dataset.split('\\')[-1].split('/')[-1]
        datasets_bar.set_description(f"Dataset name: {dataset_name}")

        f_bar = tqdm(runs, position=1, leave=False, dynamic_ncols=True)
        for (name, f) in f_bar:
            f_bar.set_description(f"Algorithm: {name}")
            f(dataset, destination_folder)


def run_CenterParTree(dataset: str, res_folder):

    df = pd.read_csv(dataset, index_col=0)

    hyperparams_name = ["max_depth", "max_nbr_clusters", "min_samples_leaf", "min_samples_split",
                        "max_nbr_values_cat", "bic_eps", "random_state", "metric_con", "metric_cat"]

    parameters = [
        [2, 3, 4, 6, 8, 10, 12],  # max_depth
        range(2, 12 + 1, 2),  # max_nbr_clusters
        [3, 30],  # range(1, 100, 30),  # min_samples_leaf
        [5, 50],  # range(2, 100, 30),  # min_samples_split
        [0], #[np.inf, 20, 100],  # max_nbr_values_cat
        np.arange(.0, .3, .1),  # bic_eps
        [42],  # random_state
        ["euclidean"],  # metric
        ["jaccard"]
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False, dynamic_ncols=True)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els])+".csv")

            filename = "CenterParTree2-" \
                       + dataset.split("/")[-1].split("\\")[-1]+"-" \
                       + ("_".join([str(x) for x in els])+".")

            if os.path.exists(res_folder+filename+"csv"):
                continue

            cpt = CenterParTree(els[0], els[1], els[2], els[3], els[4], els[5], els[6], els[7], els[8],
                                psutil.cpu_count(logical=False))

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float']))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)

            start = time.time()
            cpt.fit(X)
            stop = time.time()

            pd.DataFrame(cpt.labels_).to_csv(res_folder+filename+"zip", index=False)

            depth = 0
            leafs = 0
            nodes = 0
            for rule in cpt.get_rules():
                if rule[0]:
                    nodes += 1
                else:
                    leafs += 1

                if depth < rule[-1]:
                    depth = rule[-1]

            pd.DataFrame([[stop-start, depth, leafs, nodes] + list(els)],
                         columns=["time", "tree_depth", "tree_leafs", "tree_nodes"]+hyperparams_name)\
                .to_csv(res_folder + filename + "csv", index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els])+'.csv'}\r\n{e}")
            #raise e


def run_ImpurityParTree(dataset: str, res_folder):

    df = pd.read_csv(dataset, index_col=None)

    hyperparams_name = ["max_depth", "max_nbr_clusters", "min_samples_leaf", "min_samples_split",
                        "max_nbr_values_cat", "bic_eps", "random_state", "criteria_clf", "criteria_reg", "agg_fun"]

    parameters = [
        [2,3,4,6,8,10,12], # max_depth
        range(2, 12 +1, 2),  # max_nbr_clusters
        [3, 30],  # range(1, 100, 30),  # min_samples_leaf
        [5, 50],  # range(2, 100, 30),  # min_samples_split
        [0],  # max_nbr_values_cat
        np.arange(.0, .3, .1),  # bic_eps
        [42],  # random_state
        ["gini", "entropy", "me"],  # criteria_clf
        ["r2", "mape"], #criteria_reg
        ["mean", "min", "max"],  # agg_fun
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False, dynamic_ncols=True)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els])+".csv")

            filename = "ImpurityParTree2-" \
                       + dataset.split("/")[-1].split("\\")[-1]+"-" \
                       + ("_".join([str(x) for x in els])+".")

            if os.path.exists(res_folder+filename+"csv"):
                continue

            cpt = ImpurityParTree(els[0], els[1], els[2], els[3], els[4], els[5], els[6], els[7], els[8],
                                n_jobs=psutil.cpu_count(logical=False))

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float']))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)

            start = time.time()
            cpt.fit(X)
            stop = time.time()

            pd.DataFrame(cpt.labels_).to_csv(res_folder + filename + "zip", index=False)

            depth = 0
            leafs = 0
            nodes = 0
            for rule in cpt.get_rules():
                if rule[0]:
                    nodes += 1
                else:
                    leafs += 1

                if depth < rule[-1]:
                    depth = rule[-1]

            pd.DataFrame([[stop - start, depth, leafs, nodes] + list(els)],
                         columns=["time", "tree_depth", "tree_leafs", "tree_nodes"] + hyperparams_name) \
                .to_csv(res_folder + filename + "csv", index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els])+'.csv'}\r\n{e}")
            #raise e


def run_PrincipalParTree(dataset:str, res_folder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        df = pd.read_csv(dataset, index_col=None)

        hyperparams_name = ["max_depth", "max_nbr_clusters", "min_samples_leaf", "min_samples_split",
                            "max_nbr_values_cat", "bic_eps", "random_state", "n_components", "oblique_splits",
                            "max_oblique_features"]

        parameters = [
            [2, 3, 4, 6, 8, 10, 12],  # max_depth
            range(2, 12 + 1, 2),  # max_nbr_clusters
            [3, 30],  # range(1, 100, 30),  # min_samples_leaf
            [5, 50],  # range(2, 100, 30),  # min_samples_split
            [0],  # max_nbr_values_cat
            np.arange(.0, .3, .1),  # bic_eps
            [42],  # random_state
            [1, 2, 3],  # n_components
            [False],  # oblique_splits
            [0, 1, 2, 5]  # max_oblique_features
        ]

        els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False, dynamic_ncols=True)
        for els in els_bar:
            try:
                els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

                filename = "PrincipalParTree-" \
                           + dataset.split("/")[-1].split("\\")[-1] + "-" \
                           + ("_".join([str(x) for x in els]) + ".")

                if os.path.exists(res_folder + filename+"csv"):
                    continue

                cpt = PrincipalParTree(els[0], els[1], els[2], els[3], els[4], els[5], els[6], els[7], els[8], els[9],
                                      n_jobs=psutil.cpu_count(logical=False))

                ct = ColumnTransformer([
                    ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float']))],
                    remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

                X = ct.fit_transform(df)

                start = time.time()
                cpt.fit(X)
                stop = time.time()

                pd.DataFrame(cpt.labels_).to_csv(res_folder + filename + "zip", index=False)

                depth = 0
                leafs = 0
                nodes = 0
                for rule in cpt.get_rules():
                    if rule[0]:
                        nodes += 1
                    else:
                        leafs += 1

                    if depth < rule[-1]:
                        depth = rule[-1]

                pd.DataFrame([[stop - start, depth, leafs, nodes] + list(els)],
                             columns=["time", "tree_depth", "tree_leafs", "tree_nodes"] + hyperparams_name) \
                    .to_csv(res_folder + filename + "csv", index=False)
            except Exception as e:
                print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els])+'.csv'}\r\n{e}")
                #raise e


if __name__ == '__main__':
    datasets = [y.replace("\\", "/") for x in os.walk("../../datasets") for y in glob(os.path.join(x[0], '*.csv'))]

    run(datasets, "../../results/")
