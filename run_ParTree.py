import itertools
import os.path
import time
import warnings
import sys
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
import ParTree.algorithms.measures_utils as measures
from ParTree.classes.PrincipalParTree import PrincipalParTree


def run(datasets: list, destination_folder: str):
    runs = [
        ("PrincipalParTree", run_PrincipalParTree),
        #("CenterParTree2", run_CenterParTree),
        #("ImpurityParTree2", run_ImpurityParTree),
    ]

    datasets_bar = tqdm(datasets, position=0, leave=False)
    for dataset in datasets_bar:
        dataset_name = dataset.split('\\')[-1].split('/')[-1]
        datasets_bar.set_description(f"Dataset name: {dataset_name}")

        f_bar = tqdm(runs, position=1, leave=False)
        for (name, f) in f_bar:
            f_bar.set_description(f"Algorithm: {name}")
            f(dataset, destination_folder)


def run_CenterParTree(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    df = df.head(10)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["max_depth", "max_nbr_clusters", "min_samples_leaf", "min_samples_split", "max_nbr_values",
                        "max_nbr_values_cat", "bic_eps", "random_state", "metric_con", "metric_cat", "alfa_ind",
                        "alfa_gro", "alfa_dem", "protected_attribute"]
    parameters = [
        #[2, 3, 4, 6, 8, 10, 12],  # max_depth
        [1],
        #[len(np.unique(y))] if has_y else range(2, 12 + 1, 2),  # max_nbr_clusters
        #range(2, 12 + 1, 2),  # max_nbr_clusters
        [2],
        #[3, 30],  # range(1, 100, 30),  # min_samples_leaf
        [3],
        #[5, 50],  # range(2, 100, 30),  # min_samples_split
        [5],
        #[np.inf, 1000, 100], #[np.inf, 1000, 100],  # max_nbr_values
        [100],
        [100, 20], #[np.inf, 20, 100],  # max_nbr_values_cat
        #np.arange(.0, .3, .1),  # bic_eps
        [0.5],
        [42],  # random_state
        ["cos"],  # metric
        ["jaccard"],
        [0, 1, 2], # alfa_ind
        [0, 1, 2], # alfa_dem
        [0, 1, 2], # alfa_gro
        [4] # protected_attribute
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els])+".csv")

            colNames = hyperparams_name+["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "CenterParTree2-" \
                       + dataset.split("/")[-1].split("\\")[-1]+"-" \
                       + ("_".join([str(x) for x in els])+".csv")

            if os.path.exists(res_folder+filename):
                continue

            cpt = CenterParTree(els[0], els[1], els[2], els[3], els[4], els[5], els[6], els[7], els[8], els[9],
                                psutil.cpu_count(logical=False))

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)

            start = time.time()
            cpt.fit(X)
            stop = time.time()


            row = list(els) + [stop-start] + measures.get_metrics_uns(X, cpt.labels_)
            if has_y:
                row += measures.get_metrics_s(cpt.labels_, y)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder+filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els])+'.csv'}")
            #raise e


def run_ImpurityParTree(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["max_depth", "max_nbr_clusters", "min_samples_leaf", "min_samples_split", "max_nbr_values",
                        "max_nbr_values_cat", "bic_eps", "random_state", "criteria_clf", "criteria_reg"]

    parameters = [
        [2,3,4,6,8,10,12], # max_depth
        range(2, 12 +1, 2),  # max_nbr_clusters
        [3, 30],  # range(1, 100, 30),  # min_samples_leaf
        [5, 50],  # range(2, 100, 30),  # min_samples_split
        [np.inf, 1000, 100],  # max_nbr_values
        [20, 100],  # max_nbr_values_cat
        np.arange(.0, .3, .1),  # bic_eps
        [42],  # random_state
        ["gini", "entropy", "me"],  # criteria_clf
        ["r2", "mape"], #criteria_reg
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els])+".csv")

            colNames = hyperparams_name+["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "ImpurityParTree2-" \
                       + dataset.split("/")[-1].split("\\")[-1]+"-" \
                       + ("_".join([str(x) for x in els])+".csv")

            if os.path.exists(res_folder+filename):
                continue

            cpt = ImpurityParTree(els[0], els[1], els[2], els[3], els[4], els[5], els[6], els[7], els[8], els[9],
                                n_jobs=psutil.cpu_count(logical=False))

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)

            start = time.time()
            cpt.fit(X)
            stop = time.time()


            row = list(els) + [stop-start] + measures.get_metrics_uns(X, cpt.labels_)
            if has_y:
                row += measures.get_metrics_s(cpt.labels_, y)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder+filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els])+'.csv'}")
            #raise e


def run_PrincipalParTree(dataset:str, res_folder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        has_y = "_y.zip" in dataset

        df = pd.read_csv(dataset, index_col=None)
        df = df.head(5000)
        y = None
        # if the dataset is iris comment row below
        if has_y:
            y = df[df.columns[-1]]
            #df = df.drop(columns=[df.columns[-1]])

        hyperparams_name = ["max_depth", "max_nbr_clusters", "min_samples_leaf", "min_samples_split", "max_nbr_values",
                            "max_nbr_values_cat", "bic_eps", "random_state", "n_components", "oblique_splits",
                            "max_oblique_features", "alfa_ind", "alfa_gro", "alfa_dem", "protected_attribute"]

        parameters = [
            #[2, 3, 4, 6, 8],  # max_depth
            [2],  # max_depth
            range(2, 12 + 1, 2),  # max_nbr_clusters
            #[2]  # max_nbr_clusters
            #[3, 30],  # range(1, 100, 30),  # min_samples_leaf
            [3],  # range(1, 100, 30),  # min_samples_leaf
            [5],  # range(2, 100, 30),  # min_samples_split
            #[5, 50],  # range(2, 100, 30),  # min_samples_split
            #[np.inf, 1000, 100],  # max_nbr_values
            [100],
            #[100, 20],  # max_nbr_values_cat
            [100],
            #np.arange(.0, .3, .1),  # bic_eps
            [0.0],
            [42],  # random_state
            [1],  # n_components
            [False],  # oblique_splits
            [0, 1],  # max_oblique_features
            [0, 1, 2],  # alfa_ind
            [0, 1, 2],  # alfa_dem
            [0, 1, 2],  # alfa_gro
            [3]  # protected_attribute
        ]

        els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
        for els in els_bar:
            #try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")
            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin", "fairness_ind",
                             "fairness_dem", "fairness_gro"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score",
                                 "norm_mutual_info_score",
                                 "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "PrincipalParTree-" \
                           + dataset.split("/")[-1].split("\\")[-1] + "-" \
                           + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue


            cpt = PrincipalParTree(
                max_depth=els[0],
                max_nbr_clusters=els[1],
                min_samples_leaf=els[2],
                min_samples_split=els[3],
                max_nbr_values=els[4],
                max_nbr_values_cat=els[5],
                bic_eps=els[6],
                random_state=els[7],
                n_components=els[8],
                oblique_splits=els[9],
                max_oblique_features=els[10],
                alfa_ind=els[11],
                alfa_gro=els[12],
                alfa_dem=els[13],
                protected_attribute=els[14],
                n_jobs=psutil.cpu_count(logical=False)
            )

            protected_attribute_index = cpt.protected_attribute
            protected_attribute_name = df.columns[protected_attribute_index]
            print("\n protected_attribute_name", protected_attribute_name)
            print("\n protected_attribute_index", protected_attribute_index)

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                #('std_scaler', StandardScaler(), numerical_features),
                #("cat", OrdinalEncoder(), categorical_features)],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0
                , n_jobs=os.cpu_count()
            )

            transformed_data = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())
            protected_cols = [col for col in transformed_data.columns if col.startswith(protected_attribute_name)]
            other_cols = [col for col in transformed_data.columns if col not in protected_cols]
            new_order = other_cols[:protected_attribute_index] + protected_cols + other_cols[protected_attribute_index:]
            X = transformed_data[new_order]
            #print("Columns after changing order", X.columns)

            #if the dataset is iris comment row below
            #X = X.values[:, :-1]
            X = X.values

            start = time.time()
            cpt.fit(X)
            stop = time.time()

            row = list(els) + [stop - start] + measures.get_metrics_uns(X, cpt.labels_, protected_attribute_index)
            #print("X", X)
            #print("cpt.labels_", cpt.labels_)
            if has_y:
                row += measures.get_metrics_s(cpt.labels_, y)

            print("row", row)

            print("colNames", colNames)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
            #except Exception as e:
                #print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'}")



def get_name():
    return "ParTree"


def get_version():
    return pkg_resources.get_distribution("ParTree").version


if __name__ == '__main__':
    run(['Experiments/datasets/real/german_credit_y.zip'], 'Experiments/prova/' )

