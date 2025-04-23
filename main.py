import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from ParTree.classes.ParTree import rules_for_greenDATai
import ParTree.classes.ParTree
from ParTree.algorithms.measures_utils import get_metrics_uns, get_metrics_s
from ParTree.classes.CenterParTree import CenterParTree
from ParTree.classes.ImpurityParTree import ImpurityParTree
from ParTree.classes.ParTree import print_rules
from ParTree.classes.PrincipalParTree import PrincipalParTree
from ParTree.classes.VarianceParTree import VarianceParTree

import requests

def upload_file(filepath):
    filename = filepath.split('/')[-1].split('\\')[-1]
    url = ('https://fusion.gemma.feri.um.si/gf-test/api/ws/8a4d0d24-c969-4df9-b22d-65865c792851/'
           'services/storage/files/uc6/results/clustering/')+filename
    headers = {'x-api-key': 'c3usc9e1o6nax4rwwh77t1fbax4k1lt2'}
    with open(filepath, 'rb') as f:
        response = requests.put(url, data=f, headers=headers)
    return response.status_code, response.text

if __name__ == '__main__':
    data = pd.read_csv('Experiments/datasets/real/iris_y.zip')
    #data = pd.read_csv('Experiments/datasets/syntetic/2d-4c_y.zip')

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
    #cptree = PrincipalParTree(5, 10, 3, 5, np.inf,
    #                          np.inf, +.5, 42, 1, False, 0)
    #cptree = VarianceParTree(2, 2, 3, 5, 100, 100, 0.0, 42, 1, False)

    def cluster_info(obj):
        n_cluster = len(np.unique(obj.labels_))
        bic = "%.4f" % obj.bic_
        # print("N_cluster =", n_cluster)   , np.unique(cenptree.labels_, return_counts=True))
        # print("BIC:", bic)
        return bic, n_cluster


    scaler = StandardScaler()

    print(data.dtypes)

    ct = scaler

    X = data.values[:, :-1]
    y = data.values[:, -1]

    labels = y
    n_real_cluster = len(np.unique(y))
    X = ct.fit_transform(data[data.columns[:-1]])

    start = time.time()
    cptree.fit(X)
    end = time.time()

    r_score, adj_rand, mut_info_score, adj_mutual_info_score, norm_mutual_info_score, homog_score, complete_score, \
        v_msr_score, fwlks_mallows_score = get_metrics_s(cptree.labels_, labels)
    silhouette, calinski_harabasz, davies_bouldin = get_metrics_uns(X, cptree.labels_)
    bic, n_cluster = cluster_info(cptree)

    print(print_rules(cptree.get_rules(), X.shape[1]))

    print(rules_for_greenDATai(cptree, ct.get_feature_names_out()))

    g = ParTree.classes.ParTree.export_visualization(cptree, feature_names=ct.get_feature_names_out(), scaler=scaler)
    g.render(filename="partree_tree", format='svg', cleanup=True)

    c = ParTree.classes.ParTree.export_centroids(partree=cptree, X=X, feature_names=ct.get_feature_names_out(), scaler=None)
    print(c)

    partree = ParTree.classes.ParTree.export_centroids_svg(data.iloc[:, :-1], cptree.labels_, filename="partree_table.svg")

    print(end - start)
    print(silhouette)


    print(upload_file("partree_tree.svg"))
    print(upload_file("partree_table.svg"))




