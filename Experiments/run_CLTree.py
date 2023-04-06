import itertools
import os
import time

import numpy as np
import pandas as pd
import ParTree.algorithms.measures_utils as measures

from tqdm.auto import tqdm

from Experiments.CLTree_arff import ArffReader
from Experiments.cltree import CLTree


def run(datasets: str, destination_folder: str):
    runs = [
        ("CLTree", run_CLTree)
    ]

    datasets_bar = tqdm(datasets, position=0, leave=False)
    for dataset in datasets_bar:
        if "synthetic" not in dataset:
            continue
        dataset_name = dataset.split('\\')[-1].split('/')[-1]
        datasets_bar.set_description(f"Dataset name: {dataset_name}")

        f_bar = tqdm(runs, position=1, leave=False)
        for (name, f) in f_bar:
            f_bar.set_description(f"Algorithm: {name}")
            f(dataset, destination_folder)


def run_CLTree(dataset: str, res_folder):
    """

    """

    dataset = dataset.replace("datasets", "datasets\\arff")

    has_y = "_y.zip" in dataset

    r = ArffReader()
    data = r.read(dataset.replace(".zip", ".arff"))
    y = None
    if has_y:
        y = pd.read_csv(dataset.replace(".zip", ".csv"), index_col=0).values

    hyperparams_name = ["min_nr_instances", "min_y", "min_rd",]

    parameters = [
        [2], # min_nr_instances
        [1, 3, 5],  # min_y: The minimum number of instances a cluster must contain expressed as a percentage wrt total number of instances. Recommended value: 1-5%
        [10, 20, 30], # min_rd: Specifies whether two adjacent regions should joined to form a bigger region. Recommended value: 10-30%
    ]

    els_bar = tqdm(list(itertools.product(*parameters)), position=2, leave=False)
    for els in els_bar:
        try:
            els_bar.set_description("_".join([str(x) for x in els]) + ".csv")

            colNames = hyperparams_name + ["time", "silhouette", "calinski_harabasz", "davies_bouldin"]
            if has_y:
                colNames += ["r_score", "adj_rand", "mut_info_score", "adj_mutual_info_score", "norm_mutual_info_score",
                             "homog_score", "complete_score", "v_msr_score", "fwlks_mallows_score"]

            filename = "CLTree-" \
                       + dataset.split("/")[-1].split("\\")[-1] + "-" \
                       + ("_".join([str(x) for x in els]) + ".csv")

            if os.path.exists(res_folder + filename):
                continue

            start = time.time()
            cltree = CLTree(data)
            cltree.buildTree()
            cltree.pruneTree(els[1], els[2])
            labels = cltree.getClustersList(min_nr_instances=els[0])
            stop = time.time()

            results = []
            for i, nodo in enumerate(labels):
                for tupla in nodo.dataset.instance_values:
                    results.append((tupla[0], i))  # posizione originale nel db, cluster id


            for i in range(len(y)):
                if i in [x[0] for x in results]:
                    continue
                else:
                    results.append((i, -1))

            results.sort(key=lambda x: x[0])

            labels = [x[1] for x in sorted(results, key=lambda x: x[0])]

            row = list(els) + [stop - start] + measures.get_metrics_uns(np.array(data.instance_values.tolist())[:, 2:], labels)
            if has_y:
                row += measures.get_metrics_s(labels, y.reshape(-1))

            pd.DataFrame([row], columns=colNames).to_csv(res_folder + filename, index=False)
        except Exception as e:
            print(f"Errore dataset {dataset}, parametri {'_'.join([str(x) for x in els]) + '.csv'}")
            raise e


if __name__ == '__main__':
    run(["datasets/real/adult.zip"])


def get_name():
    return "CLTree"


def get_version():
    return "-1"
