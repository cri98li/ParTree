import time
from datetime import datetime

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
    data = pd.read_json('Experiments/datasets/real/genfair_toy.json')
    #data = pd.read_csv('Experiments/datasets/real/compas-scores-two-years_y.zip')
    #data = pd.read_csv('Experiments/datasets/syntetic/2d-4c_y.zip')
    print(data)

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
    #cptree = ImpurityParTree(n_jobs=1, max_nbr_values_cat=np.inf)
    cptree = PrincipalParTree(2, 2, 3, 5, np.inf, np.inf, 0.0, 1, 1, False, 0, mv=None, def_type="dem", protected_attribute=3)
    #cptree = VarianceParTree(2, 2, 3, 5, 100, 100, 0.0, 42, 1, False)

    def cluster_info(obj):
        n_cluster = len(np.unique(obj.labels_))
        bic = "%.4f" % obj.bic_
        #r2 = "%.4f" % obj.r2_
        return r2, n_cluster

    if cptree.def_type != "ind" and cptree.def_type != None:
        protected_attribute_index = cptree.protected_attribute
        protected_attribute_name = data.columns[protected_attribute_index]
    else:
        protected_attribute_index = None

    scaler = StandardScaler()
    #print(data.dtypes)
    ct = ColumnTransformer([
    #('std_scaler', scaler, make_column_selector(dtype_include=['int', 'float'])),
    #("cat", OrdinalEncoder(), make_column_selector(dtype_include="object")),
    #("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))
    ("cat", OneHotEncoder(), make_column_selector(dtype_include="object", pattern=f'^(?!{protected_attribute_index}$).*$'))
        ],
        remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)
    #ct = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
    #                       n_jobs=12)

    transformed_data = pd.DataFrame(ct.fit_transform(data), columns=ct.get_feature_names_out())

    if cptree.def_type == "dem" or cptree.def_type == "gro":

        # Identify columns in transformed_data that are related to the protected attribute
        protected_cols = [col for col in transformed_data.columns if col.startswith(protected_attribute_name)]

        # All columns in transformed_data that are not related to the protected attribute
        other_cols = [col for col in transformed_data.columns if col not in protected_cols]

        # Reconstruct the column order to place the protected columns at the original index
        new_order = other_cols[:protected_attribute_index] + protected_cols + other_cols[protected_attribute_index:]
        data = transformed_data[new_order]

    else:
        data = transformed_data

    #ct = ColumnTransformer([
        #('std_scaler', scaler, make_column_selector(dtype_include=['int', 'float'])),
        #("cat", OrdinalEncoder(), make_column_selector(dtype_include="object")),
        #("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))
        #("cat", OneHotEncoder(), make_column_selector(dtype_include="object", pattern=f'^(?!{protected_attribute_index}$).*$'))
    #    ],
    ##    remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)

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
    r2, n_cluster = cluster_info(cptree)

    print(end - start)
    print(silhouette)
    print(print_rules(cptree.get_rules(), X.shape[1]))

    formatted_rules = print_rules(cptree.get_rules(), X.shape[1])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"output_{timestamp}.txt"
    print("labels cptree", cptree.labels_)
    with open(filename, "a") as file:
        file.write(formatted_rules + "\n")