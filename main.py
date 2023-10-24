#prova commento
import time

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from ParTree.algorithms.measures_utils import get_metrics_uns, get_metrics_s
from ParTree.classes.CenterParTree import CenterParTree
from ParTree.classes.ImpurityParTree import ImpurityParTree
from ParTree.classes.ParTree import print_rules
from ParTree.classes.PrincipalParTree import PrincipalParTree
from ParTree.classes.VarianceParTree import VarianceParTree

if __name__ == '__main__':
    #data = pd.read_csv('Experiments/datasets/real/compas-scores-two-years.zip')
    data = pd.read_csv('Experiments/datasets/syntetic/2d-3c-no123_y.zip')

    print(data.columns)

    cptree = CenterParTree(
        max_depth=3,
        max_nbr_clusters=3,
        min_samples_leaf=3,
        min_samples_split=3,
        max_nbr_values=100,
        max_nbr_values_cat=10,
        bic_eps=0.0,
        random_state=42,
        metric_con="cos",
        metric_cat="jaccard",
        n_jobs=6,
        verbose=True
    )
    cptree = ImpurityParTree(n_jobs=1, max_nbr_values_cat=np.inf)
    #cptree = PrincipalParTree(2, 2, 3, 5, np.inf, np.inf, 0.0, 42, 1, False, 0)
    #cptree = VarianceParTree(2, 2, 3, 5, 100, 100, 0.0, 42, 1, False)

    def cluster_info(obj):
        n_cluster = len(np.unique(obj.labels_))
        bic = "%.4f" % obj.bic_
        # print("N_cluster =", n_cluster)   , np.unique(cenptree.labels_, return_counts=True))
        # print("BIC:", bic)
        return bic, n_cluster


    scaler = StandardScaler()

    print(data.dtypes)

    ct = ColumnTransformer([
        ('std_scaler', scaler, make_column_selector(dtype_include=['int', 'float'])),
        ("cat", OrdinalEncoder(), make_column_selector(dtype_include="object")),
        #("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))
        ],
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
    silhouette, calinski_harabasz, davies_bouldin = get_metrics_uns(X, cptree.labels_)
    bic, n_cluster = cluster_info(cptree)

    print(end - start)
    print(silhouette)
    print(print_rules(cptree.get_rules(), X.shape[1]))
