# %load q15_select_best_model/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q03_ohe_encoder.build import ohe_encode
from greyatomlib.multivariate_regression_project.q14_benchmarking.build import create_stats
from greyatomlib.multivariate_regression_project.q08_linear_model.build import linear_model
from greyatomlib.multivariate_regression_project.q09_advanced_model_q01_lasso.build import lasso
from greyatomlib.multivariate_regression_project.q09_advanced_model_q02_ridge.build import ridge
from greyatomlib.multivariate_regression_project.q12_feature_selection.build import feature_selection

import numpy as np
import pandas as pd

from unittest import TestCase
from inspect import getargspec

np.random.seed(9)

df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)

def complete_build(x_train, x_test, y_train, y_test):
    #Called function post label encoding
    lab_stats = create_stats(x_train, x_test, y_train, y_test,enc = 'labelencoder')
    
    #Prepare data for one hot encoding
    x_train, x_test, y_train, y_test =  split_dataset(df)
    category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']
    
    #one hot encoding
    x_train, x_test = ohe_encode(x_train, x_test, category_index)

    #Called function post one hot encoding
    ohe_stats = create_stats(x_train, x_test, y_train, y_test,enc = 'oheencoder')
    
    final_stats = pd.concat([lab_stats, ohe_stats],axis=0)
    final_stats = final_stats[['c_val', 'rmse', 'mae', 'r2']]
    
    return final_stats
        
def create_stats(x_train, x_test, y_train, y_test,enc = 'labelencoder'):
    #Run on encoded data
    lm_model, lm_y_pred,lm_stats = linear_model(x_train, x_test, y_train, y_test, 0.01)
    la_model, la_y_pred,la_stats = lasso(x_train, x_test, y_train, y_test,alpha=0.1)
    ri_model, ri_y_pred,ri_stats = ridge(x_train, x_test, y_train, y_test,alpha=0.1)

    #Filter Best feature using K percentile
    feat_sel = feature_selection(x_train,y_train,k=50)
    x_train = x_train[feat_sel]
    x_test = x_test[feat_sel]

    #Call basic model with selected features only (Linear, Lasso and Ridge)
    lm_model_fs, lm_y_pred_fs,lm_stats_fs = linear_model(x_train, x_test, y_train, y_test, 0.01)
    la_model_fs, la_y_pred_fs,la_stats_fs = lasso(x_train, x_test, y_train, y_test,alpha=0.1)
    ri_model_fs, ri_y_pred_fs,ri_stats_fs = ridge(x_train, x_test, y_train, y_test,alpha=0.1)

    #Concate the returned response.
    stats = pd.concat([lm_stats, lm_stats_fs, la_stats, la_stats_fs, ri_stats, ri_stats_fs])
    index=['lm_score_lab','lm_features_score_lab','la_score_lab','la_features_score_lab','ri_score_lab','ri_features_score_lab']

    #Rmse and mse for certain models are NaN. need to fill with zeros.
    stats[['rmse','mse']] =stats[['rmse','mse']].fillna(0)
    stats.rmse += stats.mse
    stats = stats.drop(['mse'],axis=1)
        
    return stats
    
def ohe_encode(x_train, x_test, category_index):
    #One hot encoding
    X_cat_train = x_train.iloc[:,category_index]
    x_train = x_train.drop(X_cat_train.columns, axis=1)
    x_train = pd.concat([x_train,pd.get_dummies(X_cat_train,drop_first=True)],axis=1)
    
    X_cat_test = x_test.iloc[:,category_index]
    x_test = x_test.drop(X_cat_test.columns, axis=1)
    x_test = pd.concat([x_test,pd.get_dummies(X_cat_test,drop_first=True)],axis=1)
        
    return x_train, x_test
    
#args = getargspec(complete_build)
#print len(args[0])

#stats = complete_build(x_train, x_test, y_train, y_test)
#print type(stats)
#print stats.columns
#print stats.shape
#print np.all(stats.columns == [u'c_val', u'rmse', u'mae', u'r2'])


