import numpy as np


def prepare_data(X_original, max_nbr_values, max_nbr_values_cat):
    X = np.copy(X_original)
    feature_values = dict()
    n_features = X.shape[1]
    is_categorical_feature = np.full_like(np.zeros(n_features, dtype=bool), False)
    for feature in range(n_features):
        values = np.unique(X[:, feature])
        vals = None
        if len(values) > max_nbr_values:
            _, vals = np.histogram(values, bins=max_nbr_values)
            values = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        feature_values[feature] = values

        if len(values) <= max_nbr_values_cat:
            is_categorical_feature[feature] = True

            if vals is not None:
                for original_val_idx in range(X.shape[0]):
                    for min, max, binned_val in zip(vals[:-1], vals[1:], values):
                        original_val = X[original_val_idx, feature]
                        if min < original_val and max > original_val:
                            X[original_val_idx, feature] = binned_val
                            break


    return feature_values, is_categorical_feature, X #PROBLEMA: può capitare che la feature venga binnizzata e poi dichiarata categorica.
                                                    #In questo caso il programma genera dei threshold che non esistono nei dati che causa un problema in quanto il confronto è esatto (==)