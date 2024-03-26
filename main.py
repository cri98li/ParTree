import time
from datetime import datetime
from collections import Counter
#from fairlearn.metrics import demographic_parity_difference

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

from ParTree.algorithms.measures_utils import get_metrics_uns, get_metrics_s
from ParTree.classes.CenterParTree import CenterParTree
from ParTree.classes.ImpurityParTree import ImpurityParTree
from ParTree.classes.ParTree import print_rules
from ParTree.classes.PrincipalParTree import PrincipalParTree
from ParTree.classes.VarianceParTree import VarianceParTree

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = os.path.join("logs", f"output_{timestamp}.txt")

if __name__ == '__main__':
    #data = pd.read_json('Experiments/datasets/real/genfair_toy.json')
    data = pd.read_csv('Experiments/datasets/real/compas-scores-two-years_y.zip', index_col=0)
    data.drop(columns=["r_charge_desc", "c_charge_desc"], inplace=True)

    #data = pd.read_csv('Experiments/datasets/syntetic/2d-4c_y.zip')
    #data = pd.read_csv('Experiments/datasets/real/german_credit_y.zip')
    #data = pd.read_csv('Experiments/datasets/real/bank.zip')
    #data = pd.read_csv('Experiments/datasets/real/iris_y.zip')

    print(data.columns)
    data = data.head(500)
    #data = data.iloc[:, :10]
    #index = data.columns.tolist().index('sex')
    #cptree = ImpurityParTree(n_jobs=1, max_nbr_values_cat=np.inf)
    cptree = PrincipalParTree(max_depth=2,
                              max_nbr_clusters=2,
                              min_samples_leaf=3,
                              min_samples_split=5,
                              max_nbr_values=10,
                              max_nbr_values_cat=2,
                              bic_eps=0.0,
                              random_state=42,
                              n_components=1,
                              oblique_splits=False,
                              max_oblique_features=0,
                              alfa_ind = 0,
                              alfa_gro = 0,
                              alfa_dem=1,
                              protected_attribute=2,
                              filename=filename,
                              verbose=True,
                              n_jobs=max(psutil.cpu_count(logical=False), 1))
    #cptree = VarianceParTree(2, 2, 3, 5, 100, 100, 0.0, 42, 1, False)

#    class BinaryEncoder(TransformerMixin):
#        def __init__(self):
#            self.encoder = LabelEncoder()

#        def fit(self, X, y=None):
#            self.encoder.fit(X)
#            return self

#        def transform(self, X, y=None):
#            return self.encoder.transform(X)

 #       def get_params(self, deep=True):
 #           return {}

#        def set_params(self, **params):
#            return self


#    def select_transformer(column):
#        if len(column.unique()) == 2:
#            return (f'binary_{column.name}', BinaryEncoder(), [column.name])
#        else:
#            return (f'onehot_{column.name}', OneHotEncoder(), [column.name])

    def cluster_info(obj):
        n_cluster = len(np.unique(obj.labels_))
        bic = "%.4f" % obj.bic_
        #r2 = "%.4f" % obj.r2_
        return bic, n_cluster

    #if cptree.def_type != "ind" and cptree.def_type != None:
    protected_attribute_index = cptree.protected_attribute
    protected_attribute_name = data.columns[protected_attribute_index]
    print("protected attribute name", protected_attribute_name)
    #else:
    #    protected_attribute_index = None

    scaler = StandardScaler()
    #print(data.dtypes)
    ct = ColumnTransformer([
    ('std_scaler', scaler, make_column_selector(dtype_include=['int', 'float'])),
    #("cat", OrdinalEncoder(), make_column_selector(dtype_include="object", pattern=f'^(?!{protected_attribute_index}$).*$')),
    ("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))
    #("cat", OneHotEncoder(), make_column_selector(dtype_include="object", pattern=f'^(?!{protected_attribute_index}$).*$'))
        ],
        remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)
    #ct = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
    #                       n_jobs=12)

    transformed_data = pd.DataFrame(ct.fit_transform(data), columns=ct.get_feature_names_out())
    print("COLUMN NAMES: ", ct.get_feature_names_out())

    #if cptree.def_type == "dem" or cptree.def_type == "gro":

        # Identify columns in transformed_data that are related to the protected attribute
    protected_cols = [col for col in transformed_data.columns if col.startswith(protected_attribute_name)]
        #print("protected cols", protected_cols)

        # All columns in transformed_data that are not related to the protected attribute
    other_cols = [col for col in transformed_data.columns if col not in protected_cols]

        # Reconstruct the column order to place the protected columns at the original index
    new_order = other_cols[:protected_attribute_index] + protected_cols + other_cols[protected_attribute_index:]
        #print("new order", new_order)
    data = transformed_data[new_order]
    #else:
    #data = transformed_data

    #ct = ColumnTransformer([
        #('std_scaler', scaler, make_column_selector(dtype_include=['int', 'float'])),
        #("cat", OrdinalEncoder(), make_column_selector(dtype_include="object")),
        #("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))
        #("cat", OneHotEncoder(), make_column_selector(dtype_include="object", pattern=f'^(?!{protected_attribute_index}$).*$'))
    #    ],
    ##    remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)

    #data = pd.DataFrame(ct.fit_transform(data), columns=ct.get_feature_names_out())
    #data = pd.concat([protected_attr_series.reset_index(drop=True), data.reset_index(drop=True)], axis=1)
    #data.insert(protected_attribute_index, 'protected attribute', protected_attr_series)
    #data = pd.DataFrame(ct.fit_transform(data))

    #data = data.drop(data.columns[0], axis=1)
    print("PRE COL", data.columns)
    X = data.values[:, :-1]
    #X = data.values
    print("len X", len(X))
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

    #print("labels cptree", cptree.labels_)
    with open(filename, "a") as file:
        file.write(formatted_rules + "\n")