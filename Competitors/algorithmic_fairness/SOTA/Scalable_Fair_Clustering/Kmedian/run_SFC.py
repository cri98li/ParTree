import itertools
import os
import time
from collections import Counter

from time import sleep

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from tqdm.auto import tqdm

# Import scalable clustering functions
from Competitors.algorithmic_fairness.SOTA.Scalable_Fair_Clustering.Kmedian.scalable_clustering import load_dataset, build_quadtree, tree_fairlet_decomposition, extract_clustering_labels, compute

def run_SFC_init(dataset_input, res_folder_input):
    dataset_input = 'adult'
    res_folder_input = r'C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Experiments\datasets\real\adult'
    run_SFC(dataset_input, res_folder_input)

def run_SFC(dataset_name: str, res_folder: str):
    dataset, colors = load_dataset(dataset_name)
    #dataset = pd.read_csv(r'C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Experiments\datasets\real\adult\adult_p.csv')

    # Set parameters for testing different configurations
    hyperparams_name = ["p", "q", "k"]
    parameters = [
        [1],  # Values for p
        [3],  # Values for q (should be adjusted as needed)
        [3]  # Values for k
    ]

    # Prepare for parameter iterations
    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        p, q, k = els  # Unpack the parameters

        # Update progress description
        els_bar.set_description("_".join(map(str, els)) + ".csv")

        # Setup filename for results
        #filename = f"SFC-{dataset_name}-{'_'.join(map(str, els))}.csv"
        filename = f"prova.csv"
        #if os.path.exists(res_folder + filename):
        #    continue  # Skip if results already exist

        # Prepare data (standardizing and encoding)
        ct = ColumnTransformer([
            ('std_scaler', StandardScaler(), make_column_selector(dtype_include=[np.number])),
            ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
            remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=os.cpu_count())

        #print("dataset", dataset)
        #print("dataset type", type(dataset))
        dataset = dataset.head(1000)

        X = ct.fit_transform(dataset)

        # Reset fairlet data structures
        #global FAIRLETS, FAIRLET_CENTERS
        FAIRLETS = []
        FAIRLET_CENTERS = []

        # Start fairlet decomposition and k-median clustering
        start_time = time.time()

        root = build_quadtree(X)
        #print("colors", colors)
        #cost, FAIRLETS, FAIRLET_CENTERS = tree_fairlet_decomposition(p, q, root, X, colors)
        kmedian_cost, cost, fairlet_time, total_runtime, np_idx, new_fairlets, new_centers = compute(p, q, k, X, colors, len(X), subSample=False)
        #print("FAIRLETS", FAIRLETS)
        FAIRLETS.extend(new_fairlets)
        FAIRLET_CENTERS.extend(new_centers)
        #print("FAIRLET_CENTERS", FAIRLET_CENTERS)
        overall_points = sum(len(fairlet) for fairlet in FAIRLETS)
        overall_fairlets = len(FAIRLETS)

        #print("Number of overall points:", overall_points)
        #print("Number of overall fairlets:", overall_fairlets)

        # Extract labels from FAIRLETS
        #print("passato fairlets", FAIRLETS)
        cluster_labels = extract_clustering_labels(dataset, FAIRLETS, np_idx)
        #print("cluster_labels", cluster_labels)
        cluster_counter = Counter(cluster_labels)
        for cluster, count in cluster_counter.items():
            print(f"Cluster {cluster}: {count} points")

        elapsed_time = time.time() - start_time

        # Save the results
        results = {
            'p': p,
            'q': q,
            'k': k,
            'elapsed_time': elapsed_time,
            'cost': cost,
            'cluster_labels': cluster_labels
        }

        #pd.DataFrame([results]).to_csv(res_folder + filename, index=False)


if __name__ == "__main__":
    #global FAIRLETS, FAIRLET_CENTERS
    #dataset_input = 'adult'
    #res_folder_input = r'C:\Users\fedev\Downloads\ParTree-main\ParTree-main\Experiments\datasets\real\adult'
    #res_folder_input = 'Datasets'
    run_SFC_init()
