import os
import pandas as pd

datasets = [x for x in os.walk(".")]

for dataset_path in datasets[0][2]:
    if dataset_path[-3:] == ".py" or dataset_path[-4:] == ".zip":
        continue

    print(dataset_path)

    df = pd.read_csv(dataset_path)

    df.to_csv(dataset_path[:-3]+"zip", index=False, header=False)
    os.remove(dataset_path)