# %load q15_select_best_model/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

#from greyatomlib.multivariate_regression_project.q03_ohe_encoder.build import ohe_encode

#from greyatomlib.multivariate_regression_project.q14_benchmarking.build import create_stats

from greyatomlib.multivariate_regression_project.q08_linear_model.build import linear_model
from greyatomlib.multivariate_regression_project.q12_feature_selection.build import feature_selection
from greyatomlib.multivariate_regression_project.q09_advanced_model_q01_lasso.build import lasso
from greyatomlib.multivariate_regression_project.q09_advanced_model_q02_ridge.build import ridge
from greyatomlib.multivariate_regression_project.q11_feature_selection_q02_best_k_features.build import percentile_k_features


import numpy as np
import pandas as pd

np.random.seed(9)

df = load_data('data/student-mat.csv')


x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


def complete_build(x_train, x_test, y_train, y_test):
    df1 = create_stats (x_train, x_test, y_train, y_test,'labelencoder')
    category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']
    x_train, x_test= ohe_encode(x_train,x_test,category_index=category_index)
    df2 = create_stats (x_train, x_test, y_train, y_test,'oheencoder')

    result= pd.concat([df1,df2],axis=0)
    result = result.loc[:, [ u'c_val', u'rmse', u'mae', u'r2' ]]
    return result

def ohe_encode(X_train,X_test,category_index):
    X_cat= X_train.iloc[:,category_index]
    X_train=pd.concat([X_train,pd.get_dummies(X_cat,columns=X_cat.columns )], axis=1);
    X_train=X_train.drop(X_cat.columns, axis=1)

    X_cat_test= X_test.iloc[:,category_index]
    X_test=pd.concat([X_test,pd.get_dummies(X_cat_test,columns=X_cat_test.columns )], axis=1)
    X_test=X_test.drop(X_cat_test.columns, axis=1)
    return X_train, X_test

def create_stats(X_train, X_test, y_train, y_test,enc = "labelencoder"):
    a, b,lm_score=linear_model(X_train, X_test, y_train, y_test,'')
    c, d,lasso_score=lasso(X_train, X_test, y_train, y_test)
    e, f,ridge_score=ridge(X_train, X_test, y_train, y_test)
    best_features = feature_selection(X_train,y_train, k=50)
    a, b,lm_score_bf=linear_model(X_train[best_features], X_test[best_features], y_train, y_test,'')
    c, d,lasso_score_bf=lasso(X_train[best_features], X_test[best_features], y_train, y_test)
    e, f,ridge_score_bf=ridge(X_train[best_features], X_test[best_features], y_train, y_test)

    complete_stats = pd.concat([lm_score,lasso_score,ridge_score,lm_score_bf,lasso_score_bf,ridge_score_bf],ignore_index=True)
    complete_stats.mse =complete_stats.mse.fillna(0)
    complete_stats.rmse =complete_stats.rmse.fillna(0)
    complete_stats.rmse = complete_stats.mse+ complete_stats.rmse
    del complete_stats['mse']
    return complete_stats
