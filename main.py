import time

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ParTree.classes.CenterParTree import CenterParTree
from ParTree.classes.ImpurityParTree import ImpurityParTree
from ParTree.classes.PrincipalParTree import PrincipalParTree

if __name__ == '__main__':
    data = pd.read_csv('ParTree/test/datasets/real/titanic.csv', header=0)

    #cptree = CenterParTree(n_jobs=12)
    #cptree = ImpurityParTree(n_jobs=1)
    cptree = PrincipalParTree(n_components=150)

    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)


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


    def get_metrics_s(obj, label):
        r_score = "%.4f" % rand_score(label, obj.labels_)
        adj_rand = "%.4f" % adjusted_rand_score(label, obj.labels_)
        mut_info_score = "%.4f" % mutual_info_score(label, obj.labels_)
        adj_mutual_info_score = "%.4f" % adjusted_mutual_info_score(label, obj.labels_)
        norm_mutual_info_score = "%.4f" % normalized_mutual_info_score(label, obj.labels_)
        homog_score = "%.4f" % homogeneity_score(label, obj.labels_)
        complete_score = "%.4f" % completeness_score(label, obj.labels_)
        v_msr_score = "%.4f" % v_measure_score(label, obj.labels_)
        fwlks_mallows_score = "%.4f" % fowlkes_mallows_score(label, obj.labels_)

        return r_score, adj_rand, mut_info_score, adj_mutual_info_score, norm_mutual_info_score, \
            homog_score, complete_score, v_msr_score, fwlks_mallows_score


    def get_metrics_uns(X, obj):
        try:
            silhouette = "%.4f" % silhouette_score(X, obj.labels_)
            calinski_harabasz = "%.4f" % calinski_harabasz_score(X, obj.labels_)
            davies_bouldin = "%.4f" % davies_bouldin_score(X, obj.labels_)
        except ValueError:
            silhouette = 0
            calinski_harabasz = 0
            davies_bouldin = 0
            pass

        return silhouette, calinski_harabasz, davies_bouldin


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
        ("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
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
        v_msr_score, fwlks_mallows_score = get_metrics_s(cptree, labels)
    silhouette, calinski_harabasz, davies_bouldin = get_metrics_uns(X, cptree)
    bic, n_cluster = cluster_info(cptree)

    print(end - start)
    print(silhouette)
