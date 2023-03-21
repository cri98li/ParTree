import numpy as np
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, fowlkes_mallows_score, v_measure_score, completeness_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score


def get_metrics_s(clust_id, y):
    r_score = "%.4f" % rand_score(y, clust_id)
    adj_rand = "%.4f" % adjusted_rand_score(y, clust_id)
    mut_info_score = "%.4f" % mutual_info_score(y, clust_id)
    adj_mutual_info_score = "%.4f" % adjusted_mutual_info_score(y, clust_id)
    norm_mutual_info_score = "%.4f" % normalized_mutual_info_score(y, clust_id)
    homog_score = "%.4f" % homogeneity_score(y, clust_id)
    complete_score = "%.4f" % completeness_score(y, clust_id)
    v_msr_score = "%.4f" % v_measure_score(y, clust_id)
    fwlks_mallows_score = "%.4f" % fowlkes_mallows_score(y, clust_id)

    return [r_score, adj_rand, mut_info_score, adj_mutual_info_score, norm_mutual_info_score,
            homog_score, complete_score, v_msr_score, fwlks_mallows_score]


def get_metrics_uns(X, clust_id):
    if len(np.unique(clust_id)) == 1:
        return [.0, .0, .0]

    silhouette = "%.4f" % silhouette_score(X, clust_id)
    calinski_harabasz = "%.4f" % calinski_harabasz_score(X, clust_id)
    davies_bouldin = "%.4f" % davies_bouldin_score(X, clust_id)

    return [silhouette, calinski_harabasz, davies_bouldin]
