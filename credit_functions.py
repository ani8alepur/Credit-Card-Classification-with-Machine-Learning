#this file will be added to my credit card github repository
#this can be implemented wherever found useful, make sure appropriate python libraries have been loaded before using functions

#build_model splits the x and y into training and testing, fits the model on training data, and makes predictions/shows metric reports
#all returns are necessary because they can be used for future fits
def build_model(x, y, model, **kwargs):
    if isinstance(model, BaseEstimator):
        pass
    
    x_train, x_test, y_train, y_test = split(x, y, test_size = 0.25, stratify = y, random_state = 42)
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    model_report_test = classification_report(y_test, y_test_pred)
    model_report_train = classification_report(y_train, y_train_pred)
    
    return x_train, x_test, y_train, y_test, model, model_report_test, model_report_train

#example:
#start by defining x and y
#x = people_final.drop(columns = ['GOOD_BAD'])
#y = people_final['GOOD_BAD']
#run a logistic regression with the function
#from sklearn.linear_model import LogisticRegression
#log_reg = LogisticRegression(max_iter = 2000)
#x_train, x_test, y_train, y_test, model_log, model_report_test_log, model_report_train_log = build_model(x, y, log_reg)
#print("Training report:\n", model_report_train_log)

#explanation of k-fold cross validation is given in .html file
#output will be the mean & stdev of every metric that is calculated in the cross_validate function, for both training and testing
def cross_validation(x, y, columns, model):
    steps = [('over', SMOTENC(categorical_features = columns)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    scoring = ["accuracy", "roc_auc", "precision", "recall", "f1_weighted"]
    cv = StratifiedKFold(n_splits=5)
    scores = cross_validate(pipeline, x, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score = True)
    for key, value in scores.items():
        print(key, "mean:", np.mean(value), ", std. dev:", np.std(value))
        
#hyperparameter tuning
def hyperparameter_tuning(params, model, x, y):
    from sklearn.model_selection import RandomizedSearchCV
    params = dict(params)
    model_clf = RandomizedSearchCV(model, params, cv = 5, n_jobs = -1, scoring = 'f1_weighted')
    model_clf.fit(x, y)
    best_params = model_clf.best_params_
    return model_clf, best_params

#example:
# xgb_params = {'gamma': [0.5, 1, 2.5, 5],
#               'max_depth': [2, 3, 5, 7],
#               'min_child_weight': [1, 5, 10],
#               'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
#               'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
#              } 
# xgb_hyp_clf, xgb_best_params = hyperparameter_tuning(xgb_params, model_xgb, x_oversample, y_oversample)

#for this function, the model passed through should be the one that gets returned from hyperparameter_tuning
def fit_best_model(model, x, y):
    y_train_best = model.predict(x)
    report = classification_report(y_train_best, y)
    return y_train_best, report

#example:
#y_train_best_xgb, xgb_hyp_report = fit_best_model(xgb_hyp_clf, x_train, y_train)

#modularize making a confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
def make_cm(y1, y2):
    confusion_matrix_model = confusion_matrix(y1, y2)
    cm_plot = ConfusionMatrixDisplay(confusion_matrix_model)
    cm_plot.plot()
    plt.show()

#example:
#make_cm(y_train, y_train_best_xgb)