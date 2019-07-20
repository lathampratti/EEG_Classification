import scipy.io
import argparse
import numpy as np
from utils.util import prepare_data, identify_top_features, display_top_features_count
from utils.classifiers import perform_elm, perform_svm
from utils.classifiers import perform_kNearestNeighbours
from utils.classifiers import perform_leastSquareLinearClassifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--class1", required = True, help = "A .mat file having class1's data")
    ap.add_argument("--class2", required = True, help = "A .mat file having class2's data")
    ap.add_argument("--nSessions", required = True, help = "Number of sessions for each subject")
    ap.add_argument("--field_name", required = False, default = "data", help = "Field name under which the data is available in the .mat file")
    args = vars(ap.parse_args())

    file_class1 = args["class1"]
    file_class2 = args["class2"]
    nSess = int(args["nSessions"])
    field_name = args["field_name"]

    print("\nLoading the data....")
    # Load the data
    data_class1 = scipy.io.loadmat(file_class1)
    X_class1_ = data_class1.get(field_name)
    # Remove outliers (if any)
    X_class1_outliers = [28, 29, 31, 33]
    X_class1 = np.delete(X_class1_, X_class1_outliers, axis = 0)
    
    data_class2 = scipy.io.loadmat(file_class2)
    X_class2_ = data_class2.get(field_name)
    # Remove outliers (if any)
    X_class2_outliers = [17, 20, 21, 26, 16, 27]
    X_class2 = np.delete(X_class2_, X_class2_outliers, axis = 0)
    # 16 is not an outlier - session(17) belonging to the same subject
    # is an outlier. Same is the case for 27 - its counterpart 27 is an outlier.
    # Add 16 and 27 at the end so that they become one pair.
    ndim = X_class2_[16].shape[0]
    X_class2 = np.concatenate((X_class2, X_class2_[16].reshape(1,ndim)), axis=0)
    X_class2 = np.concatenate((X_class2, X_class2_[27].reshape(1,ndim)), axis=0)

    print("\nPreparing the data for training....")
    # Prepare the data set
    Xn, yn = prepare_data(X_class1, X_class2, nSess)

    # Pick only the most significant(top) features
    top_features, tscores, pvals = identify_top_features(Xn, yn, nSess)
    Xn_t = Xn[:, top_features]
    display_top_features_count(top_features)

    # Perform least squares linear classifier
    perform_leastSquareLinearClassifier(Xn_t, yn, nSess)
    
    # Perform K-nearest neighbours classifier
    perform_kNearestNeighbours(Xn_t, yn, nSess)
    
    # Perform the ELM classification with 'sigmoid' kernel
    perform_elm(Xn_t, yn, nSess, 'sigmoid')

    # Perform the ELM classification with 'gaussian' rbf kernel
    perform_elm(Xn_t, yn, nSess, 'gaussian')

    # Perform the ELM classification with 'multiquadratic' rbf kernel
    perform_elm(Xn_t, yn, nSess, 'multiquadric')
    
    # Perform the SVM classification with 'linear' kernel
    perform_svm(Xn_t, yn, nSess, 'linear')

    # Perform the SVM classification with 'rbf' kernel
    perform_svm(Xn_t, yn, nSess, 'rbf')