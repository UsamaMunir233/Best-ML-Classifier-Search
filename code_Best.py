import argparse
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
def read_data(fname):
    data = pd.read_csv(fname)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    return X, y
def train_classifier(trial, X_train, y_train):
    classifier = trial.suggest_categorical("classifier", ["RF", "SVM", "GNB", "MNB", "BNB", "KNN", "XGB", "LBGM", "MLP", "DT", "BC", "LSCV", "PC"])
    if classifier == "RF":
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 2, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_float('max_features', 0, 1)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        n_jobs = trial.suggest_int('n_jobs', 1, 10)
        random_state = trial.suggest_int('random_state', 0, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 max_features=max_features, bootstrap=bootstrap, class_weight=class_weight,
                                 n_jobs=n_jobs, random_state=random_state)

    elif classifier == "SVM":
        C = trial.suggest_float('C', 0.1, 100)
        gamma = trial.suggest_float('gamma', 1e-5, 1)
        degree = trial.suggest_int('degree', 1, 5)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        coef0 = trial.suggest_float('coef0', -1, 1)
        shrinking = trial.suggest_categorical('shrinking', [True, False])
        tol = trial.suggest_float('tol', 1e-5, 1e-1)
        max_iter = trial.suggest_int('max_iter', -1, 1000)
        model = SVC(C=C, gamma=gamma, degree=degree, kernel=kernel, coef0=coef0,
                        shrinking=shrinking, tol=tol, max_iter=max_iter)

    elif classifier == "GNB":
        model = GaussianNB()

    elif classifier == "MNB":
        alpha = trial.suggest_float('alpha', 1e-9, 1e-3)
        fit_prior = trial.suggest_categorical('fit_prior', [True, False])
        model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

    elif classifier == "BNB":
        alpha = trial.suggest_float('alpha', 1e-9, 1e-3)
        binarize = trial.suggest_float('binarize', 0.0, 1.0)
        fit_prior = trial.suggest_categorical('fit_prior', [True, False])
        model = BernoulliNB(alpha=alpha, binarize=binarize, fit_prior=fit_prior)

    elif classifier == "KNN":
        n_neighbors = trial.suggest_int('n_neighbors', 2, 100)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        p = trial.suggest_int('p', 1, 2)
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf_size = trial.suggest_int('leaf_size', 1, 100)
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
        metric_params = trial.suggest_categorical('metric_params', [None, {}])
        n_jobs = trial.suggest_int('n_jobs', 1, 10)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p, algorithm=algorithm,leaf_size=leaf_size, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
    
    elif classifier == "XBG":
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1.0)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        max_depth = trial.suggest_int('max_depth', 2, 20)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
        subsample = trial.suggest_float('subsample', 0, 1)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0, 1)
        gamma = trial.suggest_float('gamma', 0, 1)
        reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
        reg_lambda = trial.suggest_float('reg_lambda', 0, 1)
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 100)
        objective = trial.suggest_categorical('objective', ['binary:logistic', 'multi:softmax', 'multi:softprob'])
        random_state = trial.suggest_int('random_state', 0, 100)
        model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
                        colsample_bylevel=colsample_bylevel, gamma=gamma, reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, objective=objective,
                        random_state=random_state)

    elif classifier == "LBGM":
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1.0)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        num_leaves = trial.suggest_int('num_leaves', 2, 50)
        min_child_samples = trial.suggest_int('min_child_samples', 1, 20)
        subsample = trial.suggest_float('subsample', 0, 1)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0, 1)
        reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
        reg_lambda = trial.suggest_float('reg_lambda', 0, 1)
        objective = trial.suggest_categorical('objective', ['binary', 'multiclass'])
        random_state = trial.suggest_int('random_state', 0, 100)
        model = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators, num_leaves=num_leaves,min_child_samples=min_child_samples, subsample=subsample, colsample_bytree=colsample_bytree,reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective=objective, random_state=random_state)
    
    elif classifier == "MLP":
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
        hidden_units = trial.suggest_int('hidden_units', 50, 200)
        layers = trial.suggest_int('layers', 1, 3)
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
        model = MLPClassifier(learning_rate_init=learning_rate, hidden_layer_sizes=[hidden_units] * layers, alpha=alpha, activation=activation)
    
    elif classifier == "DT":
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("max_depth", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_float("max_features", 0.1, 1.0)
        random_state = trial.suggest_int('random_state', 0, 100)
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=random_state)
    elif classifier == "BC":
        base_estimator = DecisionTreeClassifier()
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_samples = trial.suggest_float("max_samples", 0.1, 1.0)
        max_features = trial.suggest_float('max_features', 0.1, 1)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        bootstrap_features = trial.suggest_categorical("bootstrap_features", [True, False])
        n_jobs = trial.suggest_int("n_jobs", 1, 10)
        random_state = trial.suggest_int('random_state', 0, 100)
        model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, max_samples=max_samples,
                           max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features,
                           n_jobs=n_jobs, random_state=random_state)
    
    elif classifier == "LSVC":
        C = trial.suggest_float("C", 1e-5, 1e5)
        dual = trial.suggest_categorical("dual", [True, False])
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
        tol = trial.suggest_float("tol", 1e-5, 1e-1)
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        random_state = trial.suggest_int('random_state', 0, 100)
        model = LinearSVC(C=C, dual=dual, penalty=penalty, loss=loss, tol=tol, max_iter=max_iter, random_state=random_state)
    
    else: 
        classifier == "PC"
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1)
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        tol = trial.suggest_float("tol", 1e-5, 1e-1)
        penalty = trial.suggest_categorical("penalty", [None, 'l1', 'l2', 'elasticnet'])
        eta0 = trial.suggest_float("eta0", 0.1, 1.0)
        random_state = trial.suggest_int('random_state', 0, 100)
        model = Perceptron(alpha=alpha, max_iter=max_iter, tol=tol, penalty=penalty, eta0=eta0, random_state=random_state)
    
    model.fit(X_train, y_train)
    
    return model 




def evaluate_classifier(model, X_test, y_test):
    kf = KFold(n_splits=args.cv)
    y_pred = model.predict(y_test)
    accuracy = np.mean(cross_val_score(model, X_test, y_pred, cv=kf, scoring='accuracy'))
    f1 = np.mean(cross_val_score(model,X_test, y_pred, cv=kf, scoring='f1_weighted'))
    precision = np.mean(cross_val_score(model, X_test, y_pred, cv=kf, scoring='precision_macro' ))
    recall = np.mean(cross_val_score(model, X_test, y_pred, cv=kf, scoring='recall_weighted'))
    return  f1, precision, recall, accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="path to input file")
    parser.add_argument("-t", "--test_size", default=0.2, help="test size(default:0.2)")
    parser.add_argument("-cv", "--cv", type=int, default=5, help="Number of folds for cross-validation (default: 5)")
    parser.add_argument("-n", "--n_trials", type=int, default=100, help="Number of trials for the optimization(default:100)")
    parser.add_argument("-o", "--outfile", default=None, help="path to output file")
    args = parser.parse_args()
    X,y = read_data(args.file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(args.test_size), random_state=42)
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: evaluate_classifier(train_classifier(trial, X_train, y_train), X_test, y_test)[0], n_trials=args.n_trials)
best_trial = study.best_trial
best_model = train_classifier(best_trial, X_train, y_train)
accuracy, f1, precision, recall = evaluate_classifier(best_model,X_test, y_test)
print("Best model: ", best_trial.params)
print("Accuracy: ", accuracy)
print("F1 score: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)

if args.outfile:
    study_df = study.trials_dataframe()
    result = {
        "classifier": best_trial.params["classifier"],
        "Parameters": best_trial.params,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    combined_df = pd.concat([study_df, pd.DataFrame([result])])
    combined_df.to_csv(args.outfile, index=False)
else:
    print("Error in Program")