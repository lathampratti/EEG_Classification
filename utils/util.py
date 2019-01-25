import numpy as np
from scipy import stats
from functools import reduce
from sklearn.model_selection import LeaveOneGroupOut

# Identify the most signifcant features of the data
#     input: 
#        Xn - array with data from both the classes
#        yn - label
#        nSess - number of sessions per subject
#     output:
#        top_features - most signifcant features
#        top_features_tscores - tscores corresponding to those features
#        top_features_pval - p-vals corresponding to the tscores
def identify_top_features(Xn, yn, nSess=1):
    features_a = []
    tscores_a = []
    pval_a = []

    groups = get_groups(Xn, nSess)
    logo_fold = LeaveOneGroupOut()
    n_folds = logo_fold.get_n_splits(groups = groups)

    print("\nIdentify the signifcant features:")

    # Perform LOOCV to identify the most significant features
    # For each iteration sessions of one subject are left out, the
    # most signifcant features are identified using the sessions of the
    # remaining subjects.
    print(" Performing Leave one subject out cross fold(#folds = %d)" % n_folds)

    for train_index, test_index in logo_fold.split(Xn, yn, groups):
        X_train, X_test = Xn[train_index], Xn[test_index]
        y_train, y_test = yn[train_index], yn[test_index]
        x1_idx = np.argwhere(y_train == 0).flatten()
        x2_idx = np.argwhere(y_train == 1).flatten()
        x1 = X_train[x1_idx]
        x2 = X_train[x2_idx]
        top_features, tscore, pval = get_ttest_scores(x1, x2)
        features_a.append(top_features)
        tscores_a.append(tscore)
        pval_a.append(pval)

    # Pick the intersection of the features across all the iterations
    top_features = np.array(list(reduce(set.intersection,
                                        [set(item) for item in features_a ])))

    nfeatures = top_features.shape[0]
    top_features_tscores = np.zeros(nfeatures)
    top_features_pval = np.zeros(nfeatures)

    # Get the t-scores and p-values of the signifcant features
    iter = 0
    for tf in top_features:
        for i,v in enumerate(features_a):
            if tf in v:
                i1 = np.where(v == tf)
                top_features_tscores[iter] = tscores_a[i][i1]
                top_features_pval[iter] = pval_a[i][i1]
                iter += 1
                break

    return top_features, top_features_tscores, top_features_pval


# A subject's data has to be left out during cross validation.
# Assign a unique number to each subject (all the nSess session of a
# subject will have the same group number)
def get_groups(Xn, nSess=1):
    groups = np.zeros(Xn.shape[0])
    groups_iter = np.arange(0, len(groups), nSess)
    for i in groups_iter:
        groups[i:i+nSess] = (i / nSess)

    return groups 

# Displays the count of the top features corresponding to alpha, beta,
# gamma and, theta bands
def display_top_features_count(top_features):
    nalpha = len(top_features[top_features < 1596])
    nbeta = len(top_features[top_features < 3192]) - nalpha
    ngamma = len(top_features[top_features < 4788]) - (nalpha + nbeta)
    ntheta = len(top_features[top_features >= 4788])
    print("  Significant features:")
    print("    alpha band = %4d" % nalpha)
    print("     beta band = %4d" % nbeta)
    print("    gamma band = %4d" % ngamma)
    print("    theta band = %4d" % ntheta)
    print("         Total = %4d" % len(top_features))

# Get the ttest scores between two samples
#     input: 
#        x1 - sample1
#        x2 - sample2
#         t_threshold - threshold to choose the significant features 
#     ouput:
#        top_features - indices of the features that are signifcant
#        tscores - t-test scores of the signifcant features
#        pvals - p-values corresponding to the t-test scores
def get_ttest_scores(x1, x2, t_threshold=3):
    tscore, pval = stats.ttest_ind(x1, x2)
    tscore_abs = np.abs(tscore)
    temp = np.sort(tscore_abs)
    tscore_sort = temp[::-1]
    temp = np.argsort(tscore_abs)
    tscore_idx = temp[::-1]
    pval_sort = pval[tscore_idx]
    tscore_s = tscore[tscore_idx]
    val_with_score_gt1 = len(tscore_sort[tscore_sort > t_threshold])
    top_features = tscore_idx[0:val_with_score_gt1]
    tscores = tscore_s[0:val_with_score_gt1]
    pvals = pval_sort[0:val_with_score_gt1]
    return top_features, tscores, pvals

# Prepare the data set
# Samples from both the classes are placed alternatively
def prepare_data(X_class1, X_class2, nSess=1):
    nsamples_class1 = X_class1.shape[0]
    nsamples_class2 = X_class2.shape[0]
    ndimensions = X_class2.shape[1]

    total_samples = nsamples_class1 + nsamples_class2

    Xn = np.zeros((total_samples, ndimensions))
    yn = np.zeros(total_samples)

    j = 0
    irange = max(nsamples_class1, nsamples_class2)
    for i in np.arange(0, irange, nSess):
        c = nSess
        if i < nsamples_class1:
            Xn[j:j+c, :] = X_class1[i:i+c, :]
            yn[j:j+c] = 0
            j += 2

        if i < nsamples_class2:
            Xn[j:j+c, :] = X_class2[i:i+c, :]
            yn[j:j+c] = 1
            j += 2

    return Xn, yn

