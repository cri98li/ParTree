import itertools
import os
import time
import warnings

from ExKMC.Tree import Tree
from ShallowTree.ShallowTree import ShallowTree
from kmodes.kmodes import KModes
import numpy as np
import pandas as pd
from pyclustering.cluster.agglomerative import agglomerative
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.encoder import cluster_encoder
from pyclustering.cluster.xmeans import xmeans

import ParTree.algorithms.measures_utils as measures

import psutil
from sklearn.cluster import KMeans, BisectingKMeans, DBSCAN, OPTICS, AgglomerativeClustering, Birch
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from ParTree.algorithms.data_preparation import prepare_data


max_nbr_values = [np.inf, 1000, 100]  # max_nbr_values
max_nbr_values_cat = [20, 100]  # max_nbr_values_cat

def run(datasets: str, destination_folder: str):
    runs = [
        #("xmeans", run_pyclust_xmeans),
        #("pycl_agglomerative", run_pyclust_agglomerativeClust),
        #("skl_agglomerative", run_sklearn_agglomerativeClust),
        #("skl_kmeans", run_sklearn_kmeans),
        #("skl_optics", run_sklearn_optics),
        #("skl_dbscan", run_sklearn_dbscan),
        #("skl_birch", run_sklearn_birch),
        #("skl_bisectiong", run_sklearn_bis_kmeans),
        #("kmodes", run_kmodes),
        ("ShallowTree", run_ShallowTree),
        ("ExKMC", run_ExKMC),

    ]

    datasets_bar = tqdm(datasets, position=0, leave=False)
    for dataset in datasets_bar:
        dataset_name = dataset.split('\\')[-1].split('/')[-1]
        datasets_bar.set_description(f"Dataset name: {dataset_name}")

        f_bar = tqdm(runs, position=1, leave=False)
        for (name, f) in f_bar:
            f_bar.set_description(f"Algorithm: {name}")
            f(dataset, destination_folder)


def run_pyclust_agglomerativeClust(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["number_clusters", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        range(2, 12 + 1, 2),  # number_clusters
        max_nbr_values,  # max_nbr_values
        max_nbr_values_cat,  # max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "pyc_agglomerativeClust-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)

            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = agglomerative(X, els[0])

            start = time.time()
            cpt.process()
            stop = time.time()

            pyClusters = cpt.get_clusters()
            pyEncoding = cpt.get_cluster_encoding()
            pyEncoder = cluster_encoder(pyEncoding, pyClusters, X)
            pyLabels = pyEncoder.set_encoding(0).get_clusters()

            row = list(els) + [stop - start] + measures.get_metrics_uns(X, pyLabels)
            if has_y:
                row += measures.get_metrics_s(pyLabels, y)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'}")
            raise e


def run_pyclust_xmeans(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["amount_initial_centers", "kmax", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        range(2, 12 + 1, 2),  # number_clusters
        range(2, 12 * 2 + 1, 2 * 2),  # kmax
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "pyc_xmeans-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            initial_centers = kmeans_plusplus_initializer(X, els[0]).initialize()
            cpt = xmeans(X, initial_centers, kmax=els[1])

            start = time.time()
            cpt.process()
            stop = time.time()

            pyClusters = cpt.get_clusters()
            pyEncoding = cpt.get_cluster_encoding()
            pyEncoder = cluster_encoder(pyEncoding, pyClusters, X)
            pyLabels = pyEncoder.set_encoding(0).get_clusters()

            row = list(els) + [stop - start] + measures.get_metrics_uns(X, pyLabels)
            if has_y:
                row += measures.get_metrics_s(pyLabels, y)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'}")
            raise e


def run_sklearn_kmeans(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["n_clusters", "init", "n_init", "max_iter", "tol",
                        "verbose", "random_state", "copy", "algorithm", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        range(2, 12 + 1, 2),  # n_clusters
        ["k-means++"],  # init
        [11],  # n_init
        [100, 300, 500],  # max_iter
        [.0001],  # tol
        [False],  # verbose
        [42],  # random_state
        [True],  # copy,
        ["lloyd"],  # algorithm
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "skl_kmeans-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = KMeans(els[0], init=els[1], n_init=els[2], max_iter=els[3], tol=els[4], verbose=els[5],
                         random_state=els[6], copy_x=els[7], algorithm=els[8])

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


def run_sklearn_bis_kmeans(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["n_clusters", "init", "n_init", "max_iter", "tol",
                        "verbose", "random_state", "copy", "algorithm", "bisecting_strategy", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        range(2, 12 + 1, 2),  # n_clusters
        ["k-means++"],  # init
        [11],  # n_init
        [100, 300, 500],  # max_iter
        [.0001],  # tol
        [False],  # verbose
        [42],  # random_state
        [True],  # copy,
        ["lloyd"],  # algorithm
        ["biggest_inertia", "largest_cluster"], #bisecting_strategy
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "skl_bis_kmeans-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = BisectingKMeans(els[0], init=els[1], n_init=els[2], max_iter=els[3], tol=els[4], verbose=els[5],
                                  random_state=els[6], copy_x=els[7], algorithm=els[8], bisecting_strategy=els[9])

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

def run_kmodes(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["n_clusters", "max_iter", "init", "n_init", "verbose", "random_state", "n_jobs", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        range(2, 12 + 1, 2),  # n_clusters
        [100, 300, 500],  # max_iter
        ["Cao"],  # init
        [11],  # n_init
        [False],  # verbose
        [42],  # random_state
        [psutil.cpu_count(logical=False)],  # n_jobs,
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "kmodes-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = KModes(n_clusters=els[0], max_iter=els[1], init=els[2], n_init=els[3], verbose=els[4],
                         random_state=els[5], n_jobs=els[6])

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

def run_sklearn_dbscan(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["eps", "min_samples", "metric", "algorithm", "n_jobs", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        [.1, .25, .5, .75, 1.],  # eps
        [2, 5, 10], #min_samples
        ["euclidean", "cosine", "correlation"],  # metric
        ["auto"],  # algorithm
        [psutil.cpu_count(logical=False)],  # n_jobs,
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "skl_DBSCAN-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = DBSCAN(eps=els[0], min_samples=els[1], metric=els[2], algorithm=els[3], n_jobs=els[4])

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

def run_sklearn_optics(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["min_samples", "max_eps", "metric", "algorithm", "n_jobs", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        [2, 3, 5, 10, 20],  # min_samples
        [np.inf], #max_eps
        ["euclidean", "cosine", "correlation"],  # metric
        ["auto"],  # algorithm
        [psutil.cpu_count(logical=False)],  # n_jobs,
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "skl_OPTICS-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = OPTICS(min_samples=els[0], max_eps=els[1], metric=els[2], algorithm=els[3], n_jobs=els[4])

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

def run_sklearn_agglomerativeClust(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["n_clusters", "metric", "linkage", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        range(2, 12 + 1, 2),  # n_clusters
        ["euclidean", "cosine", "correlation"], #metric
        ["ward", "complete", "average", "single"], #linkage: single->min, complete->max
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        if els[-1] == "ward" and els[1] != "euclidean":
            continue

        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "skl_agglomerativeClust-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = AgglomerativeClustering(n_clusters=els[0], metric=els[1], linkage=els[2])

            start = time.time()
            cpt.fit(X)
            stop = time.time()

            row = list(els) + [stop - start] + measures.get_metrics_uns(X, cpt.labels_)
            if has_y:
                row += measures.get_metrics_s(cpt.labels_, y)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'} \n\t{e}")
            #raise e


def run_sklearn_birch(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["threshold", "branching_factor", "n_clusters", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        [.5], #threshold
        [50], #branching_factor
        range(2, 12 + 1, 2),  # n_clusters
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "skl_birch-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = Birch(threshold=els[0], branching_factor=els[1], n_clusters=els[2])

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


def run_ShallowTree(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["n_clusters", "depth_factor", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        range(2, 12 + 1, 2),  # n_clusters
        [0.01, 0.03, 0.05, 0, 1],  # depth_factor
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "ShallowTree-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = ShallowTree(k=els[0], depth_factor=els[1], random_state=42)

            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)

                start = time.time()
                labels = cpt.fit_predict(X)
                stop = time.time()

            row = list(els) + [stop - start] + measures.get_metrics_uns(X, labels)
            if has_y:
                row += measures.get_metrics_s(labels, y)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'}")
            raise e


def run_ExKMC(dataset: str, res_folder):
    has_y = "_y.zip" in dataset

    df = pd.read_csv(dataset, index_col=None)
    y = None
    if has_y:
        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])

    hyperparams_name = ["n_clusters", "max_leaves", "base_tree", "max_nbr_values", "max_nbr_values_cat"]

    parameters = [
        range(2, 12 + 1, 2),  # k
        [2, 3, 4, 6, 8, 10, 12],  # max_leaves
        ["IMM", "NONE"], #base_tree
        max_nbr_values,
        max_nbr_values_cat
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        if els[0] > els[1]:
            continue

        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "ExKMC-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            ct = ColumnTransformer([
                ('std_scaler', StandardScaler(), make_column_selector(dtype_include=['int', 'float'])),
                ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
                remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

            X = ct.fit_transform(df)
            _, _, X = prepare_data(X, els[-2], els[-1])

            cpt = Tree(k=els[0], max_leaves=els[1], base_tree=els[2],  random_state=42)

            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)

                start = time.time()
                labels = cpt.fit_predict(X)
                stop = time.time()

            row = list(els) + [stop - start] + measures.get_metrics_uns(X, labels)
            if has_y:
                row += measures.get_metrics_s(labels, y)

            pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'}")
            raise e




if __name__ == '__main__':
    pass
    #TODO: test


def get_name():
    return "kmeans+dt"


def get_version():
    return "-1"
