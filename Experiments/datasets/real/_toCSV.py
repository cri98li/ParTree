import os
from glob import glob

import pandas as pd


def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df

datasets = [x for x in os.walk(".")]

for dataset_path in datasets[0][2]:
    if dataset_path[-3:] == ".py" or dataset_path[-4:] == ".zip":
        continue

    print(dataset_path)

    df = pd.read_csv(dataset_path)

    df.to_csv(dataset_path[:-3]+"zip", index=False, header=False)
    os.remove(dataset_path)

#missing values
datasets = [y for x in os.walk(".") for y in glob(os.path.join(x[0], '*.zip'))]

for dataset_path in datasets:

    df = pd.read_csv(dataset_path, header=None, index_col=None)

    if df.isna().sum().sum() == 0:
        continue

    print(f"correctiong \t{dataset_path}")

    remove_missing_values(df).to_csv(dataset_path, index=False, header=False)