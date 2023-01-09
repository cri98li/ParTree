import os
from glob import glob
import pandas as pd


#standardize format
datasets = [x for x in os.walk(".")]

for dataset_path in datasets[0][2]:
    if dataset_path[-3:] == ".py" or dataset_path[-4:] == ".zip":
        continue

    print(dataset_path)

    print(df.head()) #breakpoint: df = pd.read_csv(dataset_path, header=None, index_col=None, sep="   ")
    print(df.dtypes)

    print(f"STOP")

    df.to_csv(dataset_path[:-3]+"zip", index=False, header=False)
    os.remove(dataset_path)

#target label column
datasets = [y for x in os.walk(".") for y in glob(os.path.join(x[0], '*.pa.zip'))]

for dataset_path in datasets:

    df_pa = pd.read_csv(dataset_path, header=None, index_col=None)
    df = pd.read_csv(dataset_path.replace(".pa", ""), header=None, index_col=None)

    df[df.shape[1]] = df_pa[0]

    print(df.head())
    print(df.dtypes)

    print(f"STOP")

    df.to_csv(dataset_path.replace(".pa", ""), index=False, header=False)
    os.remove(dataset_path)
