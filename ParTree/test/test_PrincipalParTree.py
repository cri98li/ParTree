import time
import unittest
import pandas as pd
import psutil
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder

from ParTree.algorithms.measures_utils import get_metrics_uns, get_metrics_s
from ParTree.classes.ParTree import print_rules
from ParTree.classes.PrincipalParTree import PrincipalParTree



class TestPrincipalParTree(unittest.TestCase):
    def test_only_cat(self):
        df = pd.read_csv("datasets/testing_dataset.csv")

        y = df[df.columns[-1]]
        df = df.drop(columns=[df.columns[-1]])



        cl = PrincipalParTree(
            max_depth=np.inf,
            max_nbr_values_cat=np.inf,
            random_state=42,
            n_jobs=psutil.cpu_count(logical=False)
        )

        oe = OrdinalEncoder()
        ct = ColumnTransformer([
            ("cat", oe, make_column_selector(dtype_include="object"))],
            remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)

        df_ = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

        dict_names = dict()
        for name, id in zip(df["name"], df_["name"]):
            dict_names[id] = name
            print(f"{id}: \t{name}")

        n_real_cluster = len(np.unique(y))

        start = time.time()
        cl.fit(df_.values)
        end = time.time()

        print(print_rules(cl.get_rules(), df.shape[1], feature_names=df.columns))

        silhouette, calinski_harabasz, davies_bouldin = get_metrics_uns(df_.values, cl.labels_)

        r_score, adj_rand, mut_info_score, adj_mutual_info_score, norm_mutual_info_score, homog_score, complete_score, \
        v_msr_score, fwlks_mallows_score = get_metrics_s(cl.labels_, y)

        print(f"silhouette: {silhouette}, time: \t {end-start}")


        #oe.

