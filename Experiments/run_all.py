from glob import glob

import os

import run_ParTree as rpt

modules = [rpt]


def read_status():
    filename = "results/metadata_dict.txt"
    if not os.path.exists(filename):
        return dict()

    d = dict()

    with open(filename) as f:
        k, v = f.readline().split(":")
        d[k] = v

    return d


if __name__ == '__main__':
    datasets = [y for x in os.walk("datasets/") for y in glob(os.path.join(x[0], '*.zip'))]

    metadata = read_status()

    for mod in modules:
        if mod.get_name() in metadata.keys():
            if metadata[mod.get_name()] == mod.get_version():
                print(f"{mod.get_name()}:{mod.get_version()} already executed, skipping")

        try:
            mod.run(datasets, "results")
            metadata[mod.get_name()] = mod.get_version()
        except Exception as e:
            raise e

    with open("results/metadata_dict.txt", "w+") as f:
        for k, v in metadata.items():
            f.write(f"{k}:{v}\n")


