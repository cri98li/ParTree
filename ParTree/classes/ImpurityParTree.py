import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tqdm.auto import tqdm

from ParTree.algorithms.bic_estimator import bic
from ParTree.algorithms.data_splitter import DecisionSplit
from ParTree.classes.ParTree import ParTree


def gini(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0.0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0.0

    gin = 0.0
    for p in probs:
        gin += p * p

    return 2 * (1.0 - gin)


def entropy(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0.0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0.0

    ent = 0.0
    for p in probs:
        ent -= p * np.log2(p)

    return ent


def classification_error(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0.0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0.0

    return 2 * (1.0 - np.max(probs))


def r2_relu(y_true, y_pred):
    if len(y_true) < 2:
        return .0
    return max(0.0, r2_score(y_true, y_pred))


def mape_relu(y_true, y_pred):
    return min(1.0, mean_absolute_percentage_error(y_true, y_pred))


CRITERIA_CLF = {"gini": gini, "entropy": entropy, "me": classification_error}

CRITERIA_REG = {
    "r2": r2_relu,
    "mape": mape_relu,
}


class ImpurityParTree(ParTree):
    def __init__(
            self,
            max_depth=3,
            max_nbr_clusters=10,
            min_samples_leaf=3,
            min_samples_split=5,
            max_nbr_values=100,
            max_nbr_values_cat=10,
            bic_eps=0.0,
            random_state=None,
            criteria_clf="entropy",
            criteria_reg="r2",
            agg_fun=np.mean,
            n_jobs=1,
            verbose = False
    ):
        """
        :param criteria_clf:
            The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and
            "entropy" for the Shannon information gain.

        :param criteria_reg:
            The function to measure the quality of a split. Supported criteria are "r2" for the mean squared error and
            "mape" for the mean absolute percentage error.

        :param agg_fun:
            The function to aggregate impurities. Supported functions are: np.mean, np.min and np.max

        """
        super().__init__(
            max_depth,
            max_nbr_clusters,
            min_samples_leaf,
            min_samples_split,
            max_nbr_values,
            max_nbr_values_cat,
            bic_eps,
            random_state,
            n_jobs,
            verbose
        )
        self.criteria_clf = criteria_clf
        self.criteria_reg = criteria_reg
        self.agg_fun = agg_fun

    def _make_split(self, idx_iter):
        n_features = self.X.shape[1]

        best_feature = None
        best_threshold = None
        best_impurity = np.inf

        results = []

        for n in tqdm(range(n_features), position=0, leave=False, disable=not self.verbose):
            for feature in tqdm(self.feature_values[n], position=1, leave=False, disable=not self.verbose):
                results.append(self.processPoolExecutor.submit(_make_split_innerloop,
                                                               self.X,
                                                               self.criteria_clf,
                                                               self.criteria_reg,
                                                               self.agg_fun,
                                                               self.is_categorical_feature,
                                                               idx_iter,
                                                               n,
                                                               feature))

        for res in tqdm(results, disable=not self.verbose):
            best_returned_impurity, best_returned_feature, best_returned_threshold = res.result()

            if best_returned_impurity < best_impurity:
                best_feature = best_returned_feature
                best_threshold = best_returned_threshold
                best_impurity = best_returned_impurity

        if best_feature is None:
            return None, np.zeros(len(self.X[idx_iter])), np.inf, False

        clf = DecisionSplit(best_feature, best_threshold, self.is_categorical_feature[best_feature])
        labels = clf.apply(self.X[idx_iter])
        bic_children = bic(self.X[idx_iter], (np.array(labels) - 1).tolist())
        is_oblique = False

        return clf, labels, bic_children, is_oblique


def _make_split_innerloop(X, criteria_clf, criteria_reg, agg_fun, is_categorical_feature, idx_iter, feature,
                          threshold, X_perc=.2, min_X=1000):

    n_features = X.shape[1]

    if not is_categorical_feature[feature]:  # splitting feature is continuous
        cond = X[idx_iter, feature] <= threshold
    else:  # splitting feature is categorical
        cond = X[idx_iter, feature] == threshold


    X_a = X[idx_iter][cond]
    X_a = X_a[np.random.choice(X_a.shape[0], round(len(X_a) * X_perc) + 1 if round(len(X_a) * X_perc) > min_X
                else len(X_a), replace=False)]
    X_b = X[idx_iter][~cond]
    X_b = X_b[np.random.choice(X_b.shape[0], round(len(X_b) * X_perc) + 1 if round(len(X_b) * X_perc) > min_X
                else len(X_b), replace=False)]


    if len(X_a) == 0 or len(X_b) == 0:
        return [np.inf, None, None]

    impurity_list = list()

    for target_feature in range(n_features):
        if target_feature == feature:
            continue

        if is_categorical_feature[target_feature]:
            criteria = CRITERIA_CLF[criteria_clf]
            imp_a = criteria(X_a[:, target_feature])
            imp_b = criteria(X_b[:, target_feature])

        else:
            criteria = CRITERIA_REG[criteria_reg]
            mean_val_a = np.array([np.mean(X_a[:, target_feature])] * len(X_a))
            mean_val_b = np.array([np.mean(X_b[:, target_feature])] * len(X_b))
            imp_a = criteria(X_a[:, target_feature], mean_val_a)
            imp_b = criteria(X_b[:, target_feature], mean_val_b)

        impurity = len(X_a) / (len(X_a)+len(X_b)) * imp_a + len(X_b) / (len(X_a)+len(X_b)) * imp_b
        impurity_list.append(impurity)

    impurity_agg = agg_fun(impurity_list)

    return [impurity_agg, feature, threshold]
