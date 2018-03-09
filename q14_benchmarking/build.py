from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode


from greyatomlib.multivariate_regression_project.q08_linear_model.build import linear_model
from greyatomlib.multivariate_regression_project.q12_feature_selection.build import feature_selection

from greyatomlib.multivariate_regression_project.q09_advanced_model_q01_lasso.build import lasso
from greyatomlib.multivariate_regression_project.q09_advanced_model_q02_ridge.build import ridge

from greyatomlib.multivariate_regression_project.q13_plot_residuals.build import plot_residuals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(7)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


def create_stats(X_train, X_test, y_train, y_test,enc = "labelencoder"):
    _, _,linear_model_scores=linear_model(X_train, X_test, y_train, y_test,0.01)
    _, _,lasso_scores=lasso(X_train, X_test, y_train, y_test)
    _, _,ridge_scores=ridge(X_train, X_test, y_train, y_test)

    selected_features =feature_selection(X_train,y_train, k=50)
    X_train_features=X_train[selected_features]
    X_test_features=X_test[selected_features]

    _, _,linear_model_scores_features=linear_model(X_train_features, X_test_features, y_train, y_test,0.01)
    _, _,lasso_scores_features=lasso(X_train_features, X_test_features, y_train, y_test)
    _, _,ridge_scores_features=ridge(X_train_features, X_test_features, y_train, y_test)


    complete_stats = pd.concat([linear_model_scores,linear_model_scores_features,
                            #chain_stats_with_features_selection ,stats_chain,
                            lasso_scores,lasso_scores_features,ridge_scores,ridge_scores_features])
    complete_stats.index=['linear_model_scores','linear_model_scores_features',
                            #chain_stats_with_features_selection ,stats_chain,
                            'lasso_scores','lasso_scores_features','ridge_scores','ridge_scores_features']
    #complete_stats.columns=['Name', 'cross_validation','rmse','mae','r2']
    complete_stats.rmse =complete_stats.rmse.fillna(0)
    complete_stats.mse = complete_stats.mse+ complete_stats.rmse
    complete_stats = complete_stats.drop(['rmse'],axis=1)
    return complete_stats

print (create_stats(x_train,x_test, y_train, y_test))
