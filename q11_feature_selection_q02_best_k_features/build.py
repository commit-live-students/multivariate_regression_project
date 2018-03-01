# %load q11_feature_selection_q02_best_k_features/build.py
# Default imports
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
import numpy as np
import matplotlib.pyplot as plt
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
import numpy as np
import pandas as pd
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)


np.random.seed(9)
def percentile_k_features(x_train,y_train, k=50):
    model = f_regression

    skb = SelectPercentile(model, k)
    predictors = skb.fit_transform(x_train, y_train)
    scores = list(skb.scores_)
    #print scores
    top_k_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:predictors.shape[1]]
    #print top_k_index
    top_k_predictors = [x_train.columns[i] for i in top_k_index]

    return top_k_predictors

percentile_k_features(x_train,y_train,50)
