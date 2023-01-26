import time

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from ParTree.algorithms.measures_utils import get_metrics_uns, get_metrics_s
from ParTree.classes.ParTree import print_rules

import ParTree.classes.ParTree
from ParTree.classes.CenterParTree2 import CenterParTree
#from ParTree.classes.CenterParTree import CenterParTree
#from ParTree.classes.ImpurityParTree2 import ImpurityParTree
from ParTree.classes.ImpurityParTree import ImpurityParTree
from ParTree.classes.PrincipalParTree import PrincipalParTree

if __name__ == '__main__':
    #data = pd.read_csv('Experiments/datasets/real/compas-scores-two-years.zip')
    data = pd.read_csv('Experiments/datasets/real/wdbc_y.zip')

    #cptree = CenterParTree(4, 12, 3, 5, 1000, 20, 0.1, 42, "cos", "jaccard", n_jobs=12, verbose=True)
    #cptree = ImpurityParTree(n_jobs=12)
    cptree = PrincipalParTree(2, 2, 3, 5, np.inf, np.inf, 0.0, 42, 1, False, 0)

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


    def cluster_info(obj):
        n_cluster = len(np.unique(obj.labels_))
        bic = "%.4f" % obj.bic_
        # print("N_cluster =", n_cluster)   , np.unique(cenptree.labels_, return_counts=True))
        # print("BIC:", bic)
        return bic, n_cluster


    data = remove_missing_values(data)

    scaler = StandardScaler()

    ct = ColumnTransformer([
        ('std_scaler', scaler, make_column_selector(dtype_include=['int', 'float'])),
        ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object"))],
        remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)

    data = pd.DataFrame(ct.fit_transform(data), columns=ct.get_feature_names_out())

    X = data.values[:, :-1]
    y = data.values[:, -1]

    data = X
    labels = y
    n_real_cluster = len(np.unique(y))

    start = time.time()
    cptree.fit(data)
    end = time.time()

    r_score, adj_rand, mut_info_score, adj_mutual_info_score, norm_mutual_info_score, homog_score, complete_score, \
        v_msr_score, fwlks_mallows_score = get_metrics_s(cptree.labels_, labels)
    silhouette, calinski_harabasz, davies_bouldin = get_metrics_uns(X, cptree)
    bic, n_cluster = cluster_info(cptree)

    print(end - start)
    print(silhouette)
    print(print_rules(cptree.get_rules(), X.shape[1]))
