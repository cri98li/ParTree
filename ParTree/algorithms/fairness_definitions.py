import numpy as np

def _calculate_adjacency_matrix(points):
    points_arr = np.array(points)
    diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_matrix, np.nan)
    mean_distance = np.nanmean(dist_matrix)
    adjacency_matrix = (dist_matrix < mean_distance).astype(int)
    return adjacency_matrix


def _fairness_ind(n, points, labels):
    # Provare con kdtree (magari mean_distance si può stimare, o potremmo modificare la definizione di
    # individual fairness guardando alle proprietà di kdtree
    count_comp = 0
    penalties = 0
    adjacency_matrix = _calculate_adjacency_matrix(points)
    for i in range(n):
        cluster_indices = np.where(labels == labels[i])[0]
        similar_count = np.sum(adjacency_matrix[i, cluster_indices]) - adjacency_matrix[i, i]
        penalty = similar_count / len(cluster_indices)
        count_comp += 1
        penalties += penalty

    penalty = np.sum(penalties) / count_comp
    return penalty


def _fairness_dem(points, labels, cluster_indices, protected_attribute):
    penalties = 0
    count_comp = 0
    positive_prediction_rates = {}

    # we take each value of the protected attribute
    for cluster in set(labels):
        X_filtered = points[cluster_indices[cluster]]
        total_in_group = X_filtered.shape[0]
        for group in np.unique(points[:, protected_attribute]):
            positive_predictions_in_group = (X_filtered[:, protected_attribute] == group).sum()
            positive_prediction_rate = positive_predictions_in_group / total_in_group
            positive_prediction_rates[group] = positive_prediction_rate

        keys = list(positive_prediction_rates.keys())

        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1:]:
                difference = abs(positive_prediction_rates[key1] - positive_prediction_rates[key2])
                penalties += difference
                count_comp += 1

    penalty = penalties / count_comp
    return penalty


def _fairness_gro(points, labels, protected_attribute):
    group_cluster_counts = {}
    group_counts = {}
    total_cluster = {}
    count_comp = 0
    penalties = 0

    for group in np.unique(points[:, protected_attribute]):
        # group_counts[group] = (self.X[:, self.protected_attribute] == group).sum()
        group_counts[group] = (points[:, protected_attribute] == group).sum()
        for label in np.unique(labels):
            total_cluster[label] = (labels == label).sum()
            group_cluster_counts[(group, label)] = (
                    (points[:, protected_attribute] == group) & (labels == label)).sum()

    total_count = points.shape[0]

    for group, total_in_group in group_counts.items():
        # CRI: la divisione per zero quando può capitare?
        if total_in_group > 0:  # Prevent division by zero
            tot_probability = total_in_group / total_count
            for label, value in total_cluster.items():
                group_cluster_count = group_cluster_counts[(group, label)]
                group_probability = group_cluster_count / value
                diff = abs(tot_probability - group_probability)
                penalties += diff
                count_comp += 1

    penalty = np.sum(penalties) / len(np.unique(labels))
    return penalty