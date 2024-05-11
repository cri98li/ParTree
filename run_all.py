
import os
from glob import glob

from Competitors.algorithmic_fairness.SOTA.Scalable_Fair_Clustering.Kmedian.run_SFC import run_SFC_init
from Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian import run_VFC
from Competitors.algorithmic_fairness.SOTA.Scalable_Fair_Clustering.Kmedian import run_SFC
from Competitors.algorithmic_fairness.SOTA.Variational_Fair_Clustering.Kmeans_Kmedian.run_VFC import run_VFC_init
from Competitors.algorithmic_fairness.efficient_algorithms import run_EFA
from Competitors.algorithmic_fairness.efficient_algorithms.run_EFA import run_EFA_init
from run_ParTree import *

# List with the modules imported relatively
modules = [
    run_VFC,
    run_SFC,
    run_EFA
]

def read_status():
    filename = "results/metadata_dict.txt"
    if not os.path.exists(filename):
        return dict()

    with open(filename, 'r') as file:
        d = {}
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                d[key] = value
        return d

if __name__ == '__main__':
    # Updated datasets selection
    datasets_to_include = ["german", "adult", "compas"]
    datasets = [y for x in os.walk("Experiments/datasets/real") for y in glob(os.path.join(x[0], '*.zip'))]
    datasets = [d for d in datasets if any(el in d for el in datasets_to_include)]

    print(f"Selected datasets: {datasets}")

    metadata = read_status()

    for mod in modules:
        mod_name = mod.__name__
        mod_version = getattr(mod, 'get_version', lambda: 'Unknown')()

        if mod_name in metadata:
            if metadata[mod_name] == mod_version:
                print(f"{mod_name}:{mod_version} already executed, skipping")
                continue

        try:
            #mod.run(datasets, "results/")
            run_VFC_init()
            run_EFA_init(datasets)
            run_SFC_init()
            run(datasets, 'Experiments/prova/')

            metadata[mod_name] = mod_version
        except Exception as e:
            print(f"Error running {mod_name}: {e}")

    with open("results/metadata_dict.txt", "w+") as f:
        for k, v in metadata.items():
            f.write(f"{k}:{v}\n")
