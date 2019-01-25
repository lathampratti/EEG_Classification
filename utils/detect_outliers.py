import scipy.io
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import LeaveOneGroupOut
import argparse

def checkForOutliers(Xin):
    n_samples = Xin.shape[0]
    yin = np.ones(n_samples)

    groups = np.zeros(n_samples)
    groups_iter = np.arange(0, len(groups), 2)
    for i in groups_iter:
        groups[i:i+2] = (i / 2)

    logo_fold = LeaveOneGroupOut()
    n_folds = logo_fold.get_n_splits(groups=groups)

    outliers_fraction = 0.1
    rng = np.random.RandomState(42)

    # Run IsolationForest and LocalOutlierFactor classifiers
    classifiers = {
        "Isolation Forest": IsolationForest(max_samples=n_samples-2,
                                            contamination=outliers_fraction,
                                            random_state=rng),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=35,
                                                   contamination=outliers_fraction)
    }

    folds_iter_if = 0
    outlier_list_if = np.zeros((n_folds, 5))
    folds_iter_lof = 0
    outlier_list_lof = np.zeros((n_folds, 5))

    # Perform LeaveOneOutCrossFold and identify the outliers
    for train_index, outlier_index in logo_fold.split(Xin, yin, groups):
        X_train = Xin[train_index]
        y_train = yin[train_index]
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            if clf_name == "Local Outlier Factor":
                y_pred = clf.fit_predict(X_train)
                n_errors = (y_pred != y_train).sum()
                outliers_idx = np.argsort(y_pred)[0:n_errors]
                outlier_list_lof[folds_iter_lof] = outliers_idx
                folds_iter_lof += 1
            else:
                clf = clf.fit(X_train)
                y_pred = clf.predict(X_train)
                n_errors = (y_pred != y_train).sum()
                outliers_idx = np.argsort(y_pred)[0:n_errors]
                outlier_list_if[folds_iter_if] = outliers_idx
                folds_iter_if += 1

    print('\nLocal Outlier Factor:')
    print(outlier_list_lof)

    print('\nIsolation Forest:')
    print(outlier_list_if)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--class1", required = True, help = "A .mat file having class1's data")
    ap.add_argument("--class2", required = True, help = "A .mat file having class2's data")
    ap.add_argument("--field_name", required = False, default = "data", help = "Field name under which the data is available in the .mat file")
    args = vars(ap.parse_args())

    file_class1 = args["class1"]
    file_class2 = args["class2"]
    field_name = args["field_name"]

    # Load the data and check for outliers
    data_class1 = scipy.io.loadmat(file_class1)
    X_class1 = data_class1.get(field_name)
    print('\n\nClass1:')
    checkForOutliers(X_class1)

    data_class2 = scipy.io.loadmat(file_class2)
    X_class2 = data_class2.get(field_name)
    print('\n\nfile_class2:')
    checkForOutliers(X_class2)
