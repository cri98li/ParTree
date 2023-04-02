from glob import glob

import os

import run_ParTree as rpt
import run_kmeans_DT as rkd
import run_classicClustering as rcc

modules = [
    rcc,
    rkd,
    #rpt
]


def read_status():
    filename = "results/metadata_dict.txt"
    if not os.path.exists(filename):
        return dict()

    d = dict()



    return d


if __name__ == '__main__':
    datasets = [y for x in os.walk("datasets/") for y in glob(os.path.join(x[0], '*.zip'))]
    daescludere = ["adult", "churn", "compas", "fico", "german"]
    for el in daescludere:
        for i in range(len(datasets)):
            if el in datasets[i]:
                datasets.remove(datasets[i])
                break

    print(datasets)

    metadata = read_status()

    for mod in modules:
        if mod.get_name() in metadata.keys():
            if metadata[mod.get_name()] == mod.get_version():
                print(f"{mod.get_name()}:{mod.get_version()} already executed, skipping")

        try:
            mod.run(datasets, "results/")
            metadata[mod.get_name()] = mod.get_version()
        except Exception as e:
            raise e

    with open("results/metadata_dict.txt", "w+") as f:
        for k, v in metadata.items():
            f.write(f"{k}:{v}\n")


