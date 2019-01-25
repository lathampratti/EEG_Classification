import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.metrics import roc_auc_score
from utils.util import get_groups

def perform_leastSquareLinearClassifier(Xn, yn, nSess=1):
    groups = get_groups(Xn, nSess)
    logo_fold = LeaveOneGroupOut()
    n_folds = logo_fold.get_n_splits(groups = groups)

    total_samples = Xn.shape[0]
    actual_ = np.zeros((n_folds, 2))
    predict_ = np.zeros((n_folds, 2))
    decifunc_gri = np.zeros((n_folds, 2))
    folds_iter = 0

    print("\nClassify using Linear Classifier:")
    print(" Performing leave one subject out cross fold with %d outer_folds"
          % (n_folds))

    linearReg = linear_model.LinearRegression(normalize=True)
    for train_index, test_index in logo_fold.split(Xn, yn, groups):
        # X_t_test and y_test are used for calculating classifier
        # accuracy for this iteration
        X_t_train, X_t_test = Xn[train_index], Xn[test_index]
        y_train, y_test = yn[train_index], yn[test_index]
        linearReg.fit(X_t_train, y_train)
        pred_ = linearReg.predict(X_t_test)
        predict_[folds_iter] = pred_[:]>0
        decifunc_gri[folds_iter] = linearReg._decision_function(X_t_test)
        actual_[folds_iter] = y_test
        folds_iter += 1

    # Calculate the accuracy of the classifier
    actual = actual_.reshape(total_samples,)
    predict = predict_.reshape(total_samples,)
    success = (actual == predict)
    n_success = len(success[success == True])
    print(' Classification accuracy =', (n_success / (total_samples)) * 100, "%")
    print(' Confusion Matrix:\n', confusion_matrix(actual, predict))
    decifunc_gri = decifunc_gri.reshape(total_samples,)
    print(' roc_auc_score =', roc_auc_score(actual, decifunc_gri))

def perform_kNearestNeighbours(Xn, yn, nSess=1):
    groups = get_groups(Xn, nSess)
    logo_fold = LeaveOneGroupOut()
    n_folds = logo_fold.get_n_splits(groups = groups)

    total_samples = Xn.shape[0]
    n_young_samples = int(total_samples/2)
    actual_ = np.zeros((n_folds, 2))
    predict_ = np.zeros((n_folds, 2))
    decifunc = np.zeros((n_folds, 2, 2))
    ylabel = np.zeros((n_folds, 2, 2))
    ngood = np.zeros(n_folds)
    folds_iter = 0

    print("\nClassify using K-nearest neigbours method:")
    print(" Performing leave one subject out cross fold with %d outer_folds"
          " and %d inner_folds" % (n_folds, n_folds-1))

    
    # For each iteration sessions of one subject are left out, the
    # classifier is trained with the sessions of the other subjects and,
    # the classifier is tested against the data of the left out subject.
    yn_toUse = label_binarize(yn, range(3))[:,:-1]

    kNeigh = KNeighborsClassifier(n_neighbors=1, weights='uniform', leaf_size=40)
    for train_index, test_index in logo_fold.split(Xn, yn, groups):
        # X_t_test and y_test are used for calculating classifier
        # accuracy for this iteration
        X_t_train, X_t_test = Xn[train_index], Xn[test_index]
        y_train, y_test = yn[train_index], yn[test_index]
        pgrid = { "n_neighbors": np.arange(1, n_folds, 1),
                  "leaf_size": [40, 50, 60],
                  "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                  "weights": ["uniform", "distance"],
                }
        # Inner LOOCV fold to tune the hyper parameters of the classifier
        inner_fold = LeaveOneGroupOut()
        gridclf = GridSearchCV(estimator = kNeigh, param_grid = pgrid, refit=True,
                               cv = inner_fold)
        g = gridclf.fit(X_t_train, y_train, groups = groups[train_index])
        ngood[folds_iter] = gridclf.best_params_.get('n_neighbors')
        actual_[folds_iter] = y_test
        predict_[folds_iter] = gridclf.predict(X_t_test)
        ylabel[folds_iter] = yn_toUse[test_index]
        decifunc[folds_iter] = gridclf.predict_proba(X_t_test)
        folds_iter += 1

    # Calculate the accuracy of the classifier
    actual = actual_.reshape(total_samples,)
    predict = predict_.reshape(total_samples,)
    success = (actual == predict)
    n_success = len(success[success == True])
    print(" Classification accuracy =", (n_success / total_samples) * 100, "%")
    print(' Confusion Matrix:\n', confusion_matrix(actual, predict))
    ylabel = ylabel.reshape(total_samples, 2)
    decifunc = decifunc.reshape(total_samples, 2)
    print(' roc_auc_score =', roc_auc_score(ylabel, decifunc))

def perform_svm(Xn, yn, nSess=1, kernelType='linear'):
    groups = get_groups(Xn, nSess)
    logo_fold = LeaveOneGroupOut()
    n_folds = logo_fold.get_n_splits(groups = groups)

    total_samples = Xn.shape[0]
    n_young_samples = int(total_samples/2)
    actual_ = np.zeros((n_folds, 2))
    predict_ = np.zeros((n_folds, 2))
    scores = np.zeros(n_folds)
    decifunc_gri = np.zeros((n_folds, 2))
    cgood = np.zeros(n_folds)
    ggood = np.zeros(n_folds)
    folds_iter = 0

    svm = SVC(kernel = kernelType, class_weight = 'balanced',
              decision_function_shape = 'ovo', probability = True)
    print('\nClassify using SVM: (%s)' % kernelType)
    print(" Performing leave one subject out cross fold with %d outer_folds"
          " and %d inner_folds" % (n_folds, n_folds-1))
    
    # Even while training(tuning the hyper parameters) the classifier,
    # one more subject's data is left out for each training iteration.
    # So, two(outer and inner) LOOCV folds are run.
    folds_iter = 0
    for train_index, test_index in logo_fold.split(Xn, yn, groups):
        # X_t_test and y_test are used for calculating classifier
        # accuracy for this iteration
        X_t_train, X_t_test = Xn[train_index], Xn[test_index]
        y_train, y_test = yn[train_index], yn[test_index]
        nc = X_t_train.shape[1]
        X_t_std = np.std(X_t_train)
        gamma = 1 / (nc * X_t_std)
        a = svm.set_params(gamma = gamma)
        pgrid = { "C": [0.1, 1, 10, 1e2],
                "gamma": np.arange(0.01, 0.1, 0.01)
                }
        # Inner LOOCV fold to tune the hyper parameters of the classifier
        inner_fold = LeaveOneGroupOut()
        gridclf = GridSearchCV(estimator = svm, param_grid = pgrid, refit=True,
                               cv = inner_fold)
        g = gridclf.fit(X_t_train, y_train, groups = groups[train_index])
        cgood[folds_iter] = gridclf.best_params_.get('C')
        ggood[folds_iter] = gridclf.best_params_.get('gamma')
        scores[folds_iter] = gridclf.score(X_t_test, y_test)
        actual_[folds_iter] = y_test
        predict_[folds_iter] = gridclf.predict(X_t_test)
        decifunc_gri[folds_iter] = gridclf.decision_function(X_t_test)
        folds_iter += 1

    # Calculate the accuracy of the classifier
    actual = actual_.reshape(total_samples,)
    predict = predict_.reshape(total_samples,)
    success = (actual == predict)
    n_success = len(success[success == True])
    print(" Classification accuracy =", (n_success / total_samples) * 100, "%")
    print(' Confusion Matrix:\n', confusion_matrix(actual, predict))
    '''
    print("Mean of scores:", np.mean(scores))
    scoremax_idx = np.argmax(scores)
    print("Max. of C(score max):", cgood[scoremax_idx])
    print("Max. of gamma(score max):", ggood[scoremax_idx])
    '''
    decifunc_gri = decifunc_gri.reshape(total_samples,)
    print(' roc_auc_score =', roc_auc_score(actual, decifunc_gri))

def perform_elm(Xn, yn, nSess=1, kernelType='linear'):
    groups = get_groups(Xn, nSess)
    logo_fold = LeaveOneGroupOut()
    n_folds = logo_fold.get_n_splits(groups = groups)

    total_samples = Xn.shape[0]
    n_young_samples = int(total_samples/2)
    actual_ = np.zeros((n_folds, 2))
    predict_ = np.zeros((n_folds, 2))
    decifunc_gri = np.zeros((n_folds, 2))
    folds_iter = 0

    print('\nClassify using ELM: (%s)' % kernelType)
    print(" Performing leave one subject out cross fold with %d outer_folds"
          " and %d inner_folds" % (n_folds, n_folds-1))

    # For each iteration sessions of one subject are left out, the
    # classifier is trained with the sessions of the other subjects and,
    # the classifier is tested against the data of the left out subject.

    for train_index, test_index in logo_fold.split(Xn, yn, groups):
        X_t_train, X_t_test = Xn[train_index], Xn[test_index]
        y_train, y_test = yn[train_index], yn[test_index]
        
        inner_fold = LeaveOneGroupOut()
        pgrid = { "n_hidden": np.arange(10, 300, 10),
                "rbf_width": np.arange(0.1, 0.5, 0.05)
                }
        elmc_ = ELMClassifier(n_hidden=10, random_state=42, rbf_width=0.1, activation_func=kernelType, binarizer=LabelBinarizer(0, 1))
        gridclf = GridSearchCV(estimator = elmc_, param_grid = pgrid, refit=True,
                               cv = inner_fold)
        g = gridclf.fit(X_t_train, y_train, groups = groups[train_index])
        actual_[folds_iter] = y_test
        predict_[folds_iter] = gridclf.predict(X_t_test)
        decifunc_gri[folds_iter] = gridclf.decision_function(X_t_test).reshape(2,)
        folds_iter += 1
            
    actual = actual_.reshape(total_samples,)
    predict = predict_.reshape(total_samples,)
    success = (actual == predict)
    n_success = len(success[success == True])
    print(" Classification accuracy =", (n_success / total_samples) * 100, "%")
    print(' Confusion Matrix:\n', confusion_matrix(actual, predict))
    decifunc_gri = decifunc_gri.reshape(total_samples,)
    print(' roc_auc_score =', roc_auc_score(actual, decifunc_gri))
    