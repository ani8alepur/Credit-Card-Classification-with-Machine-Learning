#NOTE: because I did this in Jupyter notebooks, I downloaded the original .ipynb file containing my code because that showed me code output. This .py file is everything that the .ipynb file has, it'll just be easier to read when downloaded.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

#START OF EDA
#importing data and looking at it
application = pd.read_csv("/Users/_ovoani_/Downloads/131_project/application_record.csv")
credit = pd.read_csv("/Users/_ovoani_/Downloads/131_project/credit_record.csv")
application.head()
credit.head()
application['AGE'] = application['DAYS_BIRTH']*-1/365
application['YEARS_EMPLOYED'] = application['DAYS_EMPLOYED']*-1/365
application = application.drop(columns = ['DAYS_BIRTH', 'DAYS_EMPLOYED'])
#check bc of formulas
print(len(application[application['AGE'] < 0]))
print(len(application[application['YEARS_EMPLOYED'] < 0]))
application.loc[application['YEARS_EMPLOYED'] < 0, 'YEARS_EMPLOYED'] = 0
print(len(application[application['AGE'] < 0]))
print(len(application[application['YEARS_EMPLOYED'] < 0]))
#check for duplicates in application, start by looking @ dimensions
app_rows, app_cols = application.shape
print(app_rows)
print(app_cols)
#we have 438,557 rows and 18 columns in the application data set. lets do the same for the credit data set
credit_rows, credit_cols = credit.shape
print(credit_rows)
print(credit_cols)
#we have 1,058,575 rows and 3 columns in the credit data set.
application_clean = application.drop_duplicates(subset = 'ID', keep = False, inplace = False)
application_clean.info()
application_clean.to_csv("application_clean.csv")
application_corr_matrix = application_clean.corr()
plt.figure(figsize = (10,8))
sns.heatmap(application_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Variable Correlation Plot')
plt.show()
#not very surprised at the lack of correlation, most of these variables have nothing to do with each other
application_clean.dtypes

#lets create some graphs for our EDA
def create_graphs(df):
    cols = df.select_dtypes(include=['object']).columns
    for col in cols:
        plt.figure(figsize=(8, 6))
        df[col].value_counts().plot(kind='bar', color='skyblue')
        plt.title(f'Bar plot for {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()
        
create_graphs(application_clean)

#now let's switch gears and look at the credit data now that we've looked at application data
credit.dtypes
#new variables for credit csv can include duration, months on time per duration, ratio of on time months (C's) to duration,
#as well as for # of months late to duration, months where you actually had a balance (account for X)
#account for how late people are in days
credit_new = pd.DataFrame()
credit_new['DURATION'] = credit.groupby(['ID']).size()
credit_new.reset_index(inplace = True)
print(credit_new)
#credit_new is simply to show us the duration of data we have per ID
new_df = pd.DataFrame()
new_df['months'] = credit.groupby(['ID', 'STATUS']).size()
new_df.reset_index(inplace = True)
new_df2 = new_df.pivot_table(index='ID', columns='STATUS', values='months', fill_value = 0)
new_df2.reset_index(inplace = True)
print(new_df2)
#new_df2 shows us the number of months on record per status as given in the credit dataset, which we will merge with credit_new
credit_final = credit_new.merge(new_df2, on = 'ID')
print(credit_final)
cols = ['0', '1', '2', '3', '4', '5', 'C', 'X']
for col in cols:
    credit_final[col] = credit_final[col] / credit_final['DURATION']
credit_final['MONTHS_WITH_BALANCE'] = credit_final['DURATION']*(1 - credit_final['X'])
print(credit_final)
#what % of people have had months where they carry a 90+ day overdue balance (classified by 3, 4, 5)
credit_final['90+ days'] = credit_final['3'] + credit_final['4'] + credit_final['5']
print(credit_final)
print(len(credit_final[credit_final['90+ days'] > 0])) #331 people over 90 days late
credit_final['30 to 90 days'] = credit_final['1'] + credit_final['2']
print(credit_final)
print(len(credit_final[credit_final['30 to 90 days'] > 0])) #5295 people
credit_final['1 to 29 days'] = credit_final['0']
print(credit_final)
print(len(credit_final[credit_final['1 to 29 days'] > 0])) #39,980 people out of 45,985 have some slight delinquency
#customers who haven't had any balance
print(len(credit_final[credit_final['X'] == 1])) #4536 customers got the card but never used it. we will not consider them for the credit card at all.
#lets create a duration histogram to see stats on how long people used their cards for
duration_plot = credit_final.hist(column = 'DURATION', bins = 10)
plt.show(duration_plot)
print(max(credit_final['DURATION']))
print(min(credit_final['DURATION']))
print(np.mean(credit_final['DURATION']))
print(np.median(credit_final['DURATION']))
print(np.std(credit_final['DURATION']))
#create a df of only people who have a credit card balance
credit_balance = credit_final[credit_final['X'] != 1]
print(credit_balance)
#make sure there's a match between the credit df and application record
merged_df = credit_final.merge(application_clean, on = 'ID', how = 'inner')
print(merged_df)
#36457 records, still a good number.

#now let's start to dummify some of the descriptor variables now that we've merged the cleaned application set and cleaned credit set
from sklearn.preprocessing import StandardScaler, OneHotEncoder
cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
for col in cols:
    merged_df[col].fillna('not_specified', inplace = True)
    print(merged_df[col].value_counts(dropna = False))
#the above showed us that only 'NAME_HOUSING_TYPE' had missing information, which we've changed to "not specified"
#ensure rounding to nearest integer
merged_df['MONTHS_WITH_BALANCE'] = np.round(merged_df['MONTHS_WITH_BALANCE'])
print(len(merged_df[merged_df['MONTHS_WITH_BALANCE'] >= 12]))
print(len(merged_df[merged_df['MONTHS_WITH_BALANCE'] >= 6]))
print(len(merged_df[merged_df['MONTHS_WITH_BALANCE'] == 0]))
#we can ignore the 3347 people who don't have balance
people_with_balance = merged_df[merged_df['MONTHS_WITH_BALANCE'] != 0]
people_with_balance
people_with_balance['ON_TIME_TO_60_LATE'] = people_with_balance['0'] + people_with_balance['1'] + people_with_balance['C']
#dummify variables using drop_first
columns_to_dummy = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY', 'NAME_EDUCATION_TYPE','NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
                    'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE']
people_with_balance = pd.get_dummies(people_with_balance, columns = columns_to_dummy, dtype = 'int', drop_first = True)
people_with_balance
#set our good/bad condition, goods are set to 0 and bads set to 1
#in english: you're considered a good candidate (0) if your combined ratio of on time payments, 1-29 days late, and 30-59 days late is greater than 80%.
people_with_balance['GOOD_BAD'] = np.where((people_with_balance['ON_TIME_TO_60_LATE'] >= 0.8), 0, 1)
people_with_balance
#lets now scale the continuous variables we have in this dataset
columns_to_scale = ['CNT_CHILDREN','AMT_INCOME_TOTAL', 'AGE','YEARS_EMPLOYED', 'CNT_FAM_MEMBERS']
st = StandardScaler()
people_with_balance[columns_to_scale] = st.fit_transform(people_with_balance[columns_to_scale])
#lets also standardize duration and months with balance
duration_scale = ['DURATION', 'MONTHS_WITH_BALANCE']
st = StandardScaler()
people_with_balance[duration_scale] = st.fit_transform(people_with_balance[duration_scale])
people_with_balance
#we have all our variables that we want standardized and dummified for us to split them into predictor and prediction variables
#lets save people_with_balance to a csv
people_with_balance.to_csv("people.csv", index = False)
people_final = pd.read_csv("people.csv")
people_final
people_final = people_final.drop(columns = ['ID', '0', '1', '2', '3', '4', '5', 'C', 'X', '90+ days', '30 to 90 days', '1 to 29 days',
                            'FLAG_MOBIL', 'ON_TIME_TO_60_LATE', 'DURATION'])
people_final
#our data is now ready with the appropriate predictors and prediction variable
#we deleted the predictor variables we deleted because they are directly used for good/bad, so we don't want it.
#lets now save this data to a csv
people_final.to_csv("final_modeling_data.csv", index = False)

#let's start the modeling part of the project
modeling_data = pd.read_csv("final_modeling_data.csv")
modeling_data
x = modeling_data.drop(columns = ['GOOD_BAD'])
y = modeling_data['GOOD_BAD']
#getting our testing and training data, stratify on prediction variable (y)
from sklearn.model_selection import train_test_split as split
import sklearn.metrics as metrics
x_train, x_test, y_train, y_test = split(x, y, test_size = 0.2, stratify = y, random_state = 42)
#lets take a second to look at the counts of good/bad. this will inform how balanced our data is
modeling_data['GOOD_BAD'].value_counts().plot(kind='bar', color='skyblue')
plt.show()

#lets fit our data on a logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, confusion_matrix, classification_report, accuracy_score, recall_score
logistic_model = LogisticRegression(max_iter = 2000).fit(x_train, y_train)
y_log_test_pred = logistic_model.predict(x_test)
y_log_train_pred = logistic_model.predict(x_train)
print("Training accuracy: ", accuracy_score(y_train, y_log_train_pred))
print("Testing accuracy: ", accuracy_score(y_test, y_log_test_pred))
log_test_report = classification_report(y_test, y_log_test_pred)
print(log_test_report)
f1_test_logistic = f1_score(y_test, y_log_test_pred, average = 'weighted')
print("F1 Score: ", f1_test_logistic)
precision_test_logistic = precision_score(y_test, y_log_test_pred, average = 'weighted')
print("Precision Score: ", precision_test_logistic)
#lets get these same statistics for the training data
log_train_report = classification_report(y_train, y_log_train_pred)
print(log_train_report)
f1_train_logistic = f1_score(y_train, y_log_train_pred, average = 'weighted')
print("F1 Score: ", f1_train_logistic)
precision_train_logistic = precision_score(y_train, y_log_train_pred, average = 'weighted')
print("Precision Score: ", precision_train_logistic)
#lets take a look at feature importance to see which predictor variables have the highest influence
features = x.columns
coefficients = logistic_model.coef_[0]
logistic_importance = pd.DataFrame({'Feature': features, 'Importance': np.abs(coefficients)})
logistic_importance = logistic_importance.sort_values('Importance', ascending=True)
logistic_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
x.dtypes
#additionally, we can see really bad recall and f1 scores for bads (which are labeled as 1), so lets fix that by oversampling
cat_cols = []
for col in x.columns:
    if x[col].dtype == 'int64': 
        cat_cols.append(True)
    else:
        cat_cols.append(False)
print(cat_cols)
from imblearn.over_sampling import SMOTENC
smotenc = SMOTENC(categorical_features = cat_cols, random_state = 42)
x_oversample, y_oversample = smotenc.fit_resample(x_train, y_train)
y_oversample.value_counts()
logistic_model_oversample = LogisticRegression(max_iter = 1000).fit(x_oversample, y_oversample)
y_pred_test_oversample = logistic_model_oversample.predict(x_test)
y_pred_train_oversample = logistic_model_oversample.predict(x_train)
print("Training oversample accuracy: ", accuracy_score(y_train, y_pred_train_oversample))
print("Testing oversample accuracy: ", accuracy_score(y_test, y_pred_test_oversample))
logistic_oversample_report = classification_report(y_test, y_pred_test_oversample)
print(logistic_oversample_report)
f1_score_log_test = f1_score(y_test, y_pred_test_oversample, average = 'weighted')
print("F1 score: ", f1_score_log_test)
precision_log_test = precision_score(y_test, y_pred_test_oversample, average = 'weighted')
print("Precision score: ", precision_log_test)
#the oversampled data shows a worse precision score but better recall score for bads after oversampling, which is better for real-life scenarios because recall = correctly identified bads / total actual bads
logistic_train_report = classification_report(y_train, y_pred_train_oversample)
print(logistic_train_report)
f1_score_log_train = f1_score(y_train, y_pred_train_oversample, average = 'weighted')
print("F1 score: ", f1_score_log_train)
precision_log_train = precision_score(y_train, y_pred_train_oversample, average = 'weighted')
print("Precision score: ", precision_log_train)
#the purpose of k-fold cross validation is to evaluate the performance of a machine learning model.
#it splits the data into k different folds and within each folds, separates data into training and testing

#start with k folds and do SMOTE again
#https://medium.com/analytics-vidhya/how-to-carry-out-k-fold-cross-validation-on-an-imbalanced-classification-problem-6d3d942a8016
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
logistic_steps = [('over', SMOTENC(categorical_features = cat_cols)), ('model', logistic_model)]
pipeline = Pipeline(steps=logistic_steps)
cv = StratifiedKFold(n_splits=5)
scoring = ["accuracy", "roc_auc", "precision", "recall", "f1_weighted"]
scores = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score = True)
print(scores)
for key, value in scores.items():
    print("Logistic model", key, "mean:", np.mean(value), ", std. dev:", np.std(value))
#looking @ training metrics:
#accuracy = 0.6775, roc_auc = 0.7736, precision = 0.3721, f1_weighted = 0.706
#we do not need to hyperparameter tune a logistic regression because it's a simple enough model to understand and there are few hyperparameters that require tuning

#lets look at a QDA model
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda_model = QuadraticDiscriminantAnalysis().fit(x_train, y_train)
y_qda_train_pred = qda_model.predict(x_train)
y_qda_test_pred = qda_model.predict(x_test)
print("Training accuracy: ", accuracy_score(y_train, y_qda_train_pred))
print("Testing accuracy: ", accuracy_score(y_test, y_qda_test_pred))
qda_report = classification_report(y_test, y_qda_test_pred)
print(qda_report)
f1_score_qda = f1_score(y_test, y_qda_test_pred, average = 'weighted')
print("F1 Score: ", f1_score_qda)
precision_score_qda = precision_score(y_test, y_qda_test_pred, average = 'weighted')
print("Precision Score: ", precision_score_qda)
#same for training data
qda_train_report = classification_report(y_train, y_qda_train_pred)
print(qda_train_report)
f1_score_qda_train = f1_score(y_train, y_qda_train_pred, average = 'weighted')
print("F1 Score: ", f1_score_qda_train)
precision_score_qda_train = precision_score(y_train, y_qda_train_pred, average = 'weighted')
print("Precision Score: ", precision_score_qda_train)
#oversample data
qda_os = QuadraticDiscriminantAnalysis().fit(x_oversample, y_oversample)
y_qda_os_train_pred = qda_os.predict(x_train)
y_qda_os_test_pred = qda_os.predict(x_test)
print("Training oversample accuracy: ", accuracy_score(y_train, y_qda_os_train_pred))
print("Testing oversample accuracy: ", accuracy_score(y_test, y_qda_os_test_pred))
qda_os_report = classification_report(y_test, y_qda_os_test_pred)
print(qda_os_report)
f1_score_qda_os = f1_score(y_test, y_qda_os_test_pred, average = 'weighted')
print("F1 Score oversample: ", f1_score_qda_os)
precision_score_qda_os = precision_score(y_test, y_qda_os_test_pred, average = 'weighted')
print("Precision Score oversample: ", precision_score_qda_os)
#training data
qda_os_report_train = classification_report(y_train, y_qda_os_train_pred)
print(qda_os_report_train)
f1_score_qda_os = f1_score(y_train, y_qda_os_train_pred, average = 'weighted')
print("F1 Score oversample: ", f1_score_qda_os)
precision_score_qda_os = precision_score(y_train, y_qda_os_train_pred, average = 'weighted')
print("Precision Score oversample: ", precision_score_qda_os)
#k-fold cross validation for QDA
qda_steps = [('over', SMOTENC(categorical_features = cat_cols)), ('model', qda_model)]
pipeline = Pipeline(steps=qda_steps)
scoring = ["accuracy", "roc_auc", "precision", "recall", "f1_weighted"]
cv = StratifiedKFold(n_splits=5)
qda_scores = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score = True)
print(qda_scores)
for key, value in qda_scores.items():
    print("Quadratic Discriminant Analysis model", key, "mean:", np.mean(value), ", std. dev:", np.std(value))
#looking @ training metrics:
#accuracy = 0.5278, roc_auc = 0.6524, precision = 0.2745, f1_weighted = 0.5653, recall = 0.7373

#now lets look at a gradient boosted XGB model
import xgboost as xgb
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(x_train, y_train)
y_xgb_test_pred = xgb_clf.predict(x_test)
y_xgb_train_pred = xgb_clf.predict(x_train)
print("Training accuracy: ", accuracy_score(y_train, y_xgb_train_pred))
print("Testing accuracy: ", accuracy_score(y_test, y_xgb_test_pred))
xgb_report = classification_report(y_test, y_xgb_test_pred)
print(xgb_report)
f1_score_xgb = f1_score(y_test, y_xgb_test_pred, average = 'weighted')
print("F1 Score: ", f1_score_xgb)
precision_score_xgb = precision_score(y_test, y_xgb_test_pred, average = 'weighted')
print("Precision Score: ", precision_score_xgb)
#looking at training data
xgb_report_train = classification_report(y_train, y_xgb_train_pred)
print(xgb_report_train)
f1_score_xgb_train = f1_score(y_train, y_xgb_train_pred, average = 'weighted')
print("F1 Score: ", f1_score_xgb_train)
precision_score_xgb_train = precision_score(y_train, y_xgb_train_pred, average = 'weighted')
print("Precision Score: ", precision_score_xgb_train)
#now let's look at oversampled data
xgb_clf_overfit = xgb.XGBClassifier()
xgb_clf_overfit.fit(x_oversample, y_oversample)
y_os_xgb_train_pred = xgb_clf_overfit.predict(x_train)
y_os_xgb_test_pred = xgb_clf_overfit.predict(x_test)
print("Training oversampled accuracy: ", accuracy_score(y_train, y_os_xgb_train_pred))
print("Testing oversampled accuracy: ", accuracy_score(y_test, y_os_xgb_test_pred))
xgb_os_report = classification_report(y_test, y_os_xgb_test_pred)
print(xgb_os_report)
f1_score_os_xgb = f1_score(y_test, y_os_xgb_test_pred, average = 'weighted')
print("F1 Score: ", f1_score_os_xgb)
precision_score_os_xgb = precision_score(y_test, y_os_xgb_test_pred, average = 'weighted')
print("Precision Score: ", precision_score_os_xgb)
#now lets look at the training set for oversampled data
xgb_os_report_train = classification_report(y_train, y_os_xgb_train_pred)
print(xgb_os_report_train)
f1_score_os_xgb_train = f1_score(y_train, y_os_xgb_train_pred, average = 'weighted')
print("F1 Score: ", f1_score_os_xgb_train)
precision_score_os_xgb_train = precision_score(y_train, y_os_xgb_train_pred, average = 'weighted')
print("Precision Score: ", precision_score_os_xgb_train)
#k-fold cross validation
xgb_steps = [('over', SMOTENC(categorical_features = cat_cols)), ('model', xgb_clf)]
pipeline = Pipeline(steps=xgb_steps)
scoring = ["accuracy", "roc_auc", "precision", "recall", "f1_weighted"]
cv = StratifiedKFold(n_splits=5)
xgb_scores = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score = True)
print(xgb_scores)
for key, value in xgb_scores.items():
    print("XGBoost model", key, "mean:", np.mean(value), ", std. dev:", np.std(value))
#looking @ training metrics:
#accuracy = 0.8804, roc_auc = 0.9192, precision = 0.7506, f1_weighted = 0.8776

#hyperparameter tuning for XGB model
from sklearn.model_selection import RandomizedSearchCV
xgb_model = xgb.XGBClassifier()
xgb_params = {'gamma': [0.5, 1, 2.5, 5],
              'max_depth': [2, 3, 5, 7],
              'min_child_weight': [1, 5, 10],
              'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
             }
xgb_model_clf = RandomizedSearchCV(xgb_model, xgb_params, cv = 5, n_jobs = -1, scoring = "f1_weighted")
xgb_model_clf.fit(x_oversample, y_oversample)
#best parameters
xgb_best_params = xgb_model_clf.best_params_
print(xgb_best_params)
y_train_best_xgb = xgb_model_clf.predict(x_train)
print("XGB hyperparameter training accuracy: ", accuracy_score(y_train, y_train_best_xgb))
xgb_hyper_report = classification_report(y_train_best_xgb, y_train)
print(xgb_hyper_report)
f1_hyper_xgb = f1_score(y_train, y_train_best_xgb, average = 'weighted')
print("F1 hyperparameter score: ", f1_hyper_xgb)
precision_hyper_xgb = precision_score(y_train, y_train_best_xgb, average = 'weighted')
print("Precision hyperparameter score: ", precision_hyper_xgb)
#k-fold cross validation on hyperparameter tuned model
xgb_hyp_steps = [('over', SMOTENC(categorical_features = cat_cols)), ('model', xgb_model_clf)]
pipeline = Pipeline(steps=xgb_hyp_steps)
cv = StratifiedKFold(n_splits=5)
scoring = ["accuracy", "roc_auc", "precision", "recall", "f1_weighted"]
xgb_hyp_scores = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score = True)
print(xgb_hyp_scores)
for key, value in xgb_hyp_scores.items():
    print("XGBooost Hyperparameter model", key, "mean:", np.mean(value), ", std. dev:", np.std(value))
#looking @ training metrics:
#accuracy = 0.8819, roc_auc = 0.9183, precision = 0.7513, f1_weighted = 0.8794

#last model: random forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)
y_rf_train_pred = rf_clf.predict(x_train)
y_rf_test_pred = rf_clf.predict(x_test)
print("Training accuracy: ", accuracy_score(y_train, y_rf_train_pred))
print("Testing accuracy: ", accuracy_score(y_test, y_rf_test_pred))
rf_report = classification_report(y_test, y_rf_test_pred)
print(rf_report)
f1_score_rf = f1_score(y_test, y_rf_test_pred, average = 'weighted')
print("F1 Score: ", f1_score_rf)
precision_score_rf = precision_score(y_test, y_rf_test_pred, average = 'weighted')
print("Precision Score: ", precision_score_rf)
#lets do the same for training data
rf_report_train = classification_report(y_train, y_rf_train_pred)
print(rf_report_train)
f1_score_rf_train = f1_score(y_train, y_rf_train_pred, average = 'weighted')
print("F1 Score: ", f1_score_rf_train)
precision_score_rf_train = precision_score(y_train, y_rf_train_pred, average = 'weighted')
print("Precision Score: ", precision_score_rf_train)
#looking @ oversampled data
rf_oversampled_clf = RandomForestClassifier()
rf_oversampled_clf.fit(x_oversample, y_oversample)
y_rf_os_train_pred = rf_oversampled_clf.predict(x_train)
y_rf_os_test_pred = rf_oversampled_clf.predict(x_test)
print("Training oversampled accuracy: ", accuracy_score(y_train, y_rf_os_train_pred))
print("Testing oversampled accuracy: ", accuracy_score(y_test, y_rf_os_test_pred))
rf_os_report = classification_report(y_test, y_rf_os_test_pred)
print(rf_os_report)
f1_score_rf_os = f1_score(y_test, y_rf_os_test_pred, average = 'weighted')
print("F1 Score: ", f1_score_rf_os)
precision_score_rf_os = precision_score(y_test, y_rf_os_test_pred, average = 'weighted')
print("Precision Score: ", precision_score_rf_os)
#do the same for training data
rf_os_report_train = classification_report(y_train, y_rf_os_train_pred)
print(rf_os_report_train)
f1_score_rf_os_train = f1_score(y_train, y_rf_os_train_pred, average = 'weighted')
print("F1 Score: ", f1_score_rf_os_train)
precision_score_rf_os_train = precision_score(y_train, y_rf_os_train_pred, average = 'weighted')
print("Precision Score: ", precision_score_rf_os_train)
#k fold cross validation
rf_steps = [('over', SMOTENC(categorical_features = cat_cols)), ('model', rf_clf)]
pipeline = Pipeline(steps=rf_steps)
scoring = ["accuracy", "roc_auc", "precision", "recall", "f1_weighted"]
cv = StratifiedKFold(n_splits=5)
rf_scores = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score = True)
print(rf_scores)
for key, value in rf_scores.items():
    print("Random Forest model", key, "mean:", np.mean(value), ", std. dev:", np.std(value))
#looking @ training metrics:
#accuracy = 0.9899, roc_auc = 0.999, precision = 0.972, f1_weighted = 0.9899

#hyperparameter tuning for random forest
from scipy.stats import randint
rf_hyp_model = RandomForestClassifier()
rf_params = {'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [3, 5, 10, None],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': randint(1,5),
            'bootstrap': [True, False],
            }
rf_model_clf = RandomizedSearchCV(rf_hyp_model, rf_params, cv = 5, n_jobs = -1, scoring = 'f1_weighted')
rf_model_clf.fit(x_oversample, y_oversample)
print(rf_model_clf.best_params_)
y_train_best_rf = rf_model_clf.predict(x_train)
print("Random Forest hyperparameter training accuracy: ", accuracy_score(y_train, y_train_best_rf))
rf_hyper_report = classification_report(y_train_best_rf, y_train)
print(rf_hyper_report)
f1_hyper_rf = f1_score(y_train, y_train_best_rf, average = 'weighted')
print("F1 hyperparameter score: ", f1_hyper_rf)
precision_hyper_rf = precision_score(y_train, y_train_best_rf, average = 'weighted')
print("Precision hyperparameter score: ", precision_hyper_rf)
#k-fold cross validation for hyperparameter model
rf_hyp_steps = [('over', SMOTENC(categorical_features = cat_cols)), ('model', rf_model_clf)]
pipeline = Pipeline(steps=rf_hyp_steps)
scoring = ["accuracy", "roc_auc", "precision", "recall", "f1_weighted"]
cv = StratifiedKFold(n_splits=5)
rf_hyp_scores = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score = True)
print(rf_hyp_scores)
for key, value in rf_hyp_scores.items():
    print("Random Forest Hyperparameter model", key, "mean:", np.mean(value), ", std. dev:", np.std(value))
#looking @ training metrics:
#accuracy = 0.9248, roc_auc = 0.9625, precision = 0.8239, f1_weighted = 0.9281

#lets get a graph of the f1 scores after performing cross validation on the original models:
models = ['Logistic Regression', 'Naive Bayes', 'XGBoost', 'Random Forest']
f1_cv_scores = [0.706, 0.5716, 0.8776, 0.9899]
plt.bar(models, f1_cv_scores, color = "pink")
plt.xlabel("Models")
plt.ylabel("F1 Scores")
plt.title("F1 Scores for Training Data after Cross Validation")
plt.show()  

#from above, we can see our two best models are xgboost and random forest.
#does this hold true even after hyperparametrizing the models? let's find out
models = ['Naive Bayes', 'XGBoost', 'Random Forest']
f1_hyp_scores = [0.5717, 0.8794, 0.9281]
plt.bar(models, f1_hyp_scores, color = "orange")
plt.xlabel("Models")
plt.ylabel("F1 Scores")
plt.title("F1 Scores for Training Data after Hyperparametrization")
plt.show()
#and as we can see, xgboost and random forest are still the two best performing models.

#lets now fit the xgboost and random forest hyperparameter models to the testing data and make confusion matrices.
y_test_best_xgb = xgb_model_clf.predict(x_test)
print("XGB hyperparameter testing accuracy: ", accuracy_score(y_test, y_test_best_xgb))
xgb_hyper_report = classification_report(y_test_best_xgb, y_test)
print(xgb_hyper_report)
f1_hyper_xgb = f1_score(y_test, y_test_best_xgb, average = 'weighted')
print("F1 hyperparameter score: ", f1_hyper_xgb)
precision_hyper_xgb = precision_score(y_test, y_test_best_xgb, average = 'weighted')
print("Precision hyperparameter score: ", precision_hyper_xgb)
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix_xgb = confusion_matrix(y_test, y_test_best_xgb)
cm_plot_xgb = ConfusionMatrixDisplay(confusion_matrix_xgb)
cm_plot_xgb.plot()
plt.show()

y_test_best_rf = rf_model_clf.predict(x_test)
print("Random Forest hyperparameter testing accuracy: ", accuracy_score(y_test, y_test_best_rf))
rf_hyper_report = classification_report(y_test_best_rf, y_test)
print(rf_hyper_report)
f1_hyper_rf = f1_score(y_test, y_test_best_rf, average = 'weighted')
print("F1 hyperparameter score: ", f1_hyper_rf)
precision_hyper_rf = precision_score(y_test, y_test_best_rf, average = 'weighted')
print("Precision hyperparameter score: ", precision_hyper_rf)
confusion_matrix_rf = confusion_matrix(y_test, y_test_best_rf)
cm_plot_rf = ConfusionMatrixDisplay(confusion_matrix_rf)
cm_plot_rf.plot()
plt.show()

#we can see from precision and f1 score that the random forest was the best model to fit the testing data on
#we fit the testing data on the original x_test and not the oversampled data because we need to use original data for testing so that it reflects actual real-world trends
#now lets get ROC AUC for the two models
from sklearn.metrics import roc_auc_score
xgb_best_roc = roc_auc_score(y_test, y_test_best_xgb)
print("XGB Final test roc auc:", xgb_best_roc)
rf_best_roc = roc_auc_score(y_test, y_test_best_rf)
print("Random Forest Final test roc auc:", rf_best_roc) 




