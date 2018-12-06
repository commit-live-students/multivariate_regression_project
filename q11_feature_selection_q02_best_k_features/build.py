# %load q11_feature_selection_q02_best_k_features/build.py
# Default imports
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
import numpy as np
import matplotlib.pyplot as plt
import operator
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
# Write your code below
def percentile_k_features(x_train, y_train, k=50):
    Selector_f = SelectPercentile(f_regression, percentile=k)
    fs = Selector_f.fit(x_train, y_train)
    cols = list(x_train.columns)
    scores = list(fs.scores_)
    
    feature_score = dict()
    top_k_predictors = []
    for i in range(0,len(cols)):
        feature_score[cols[i]] = scores[i]
    
    sorted_x = sorted(feature_score.items(), key=operator.itemgetter(1), reverse=True)
    for obj in range(0, 16):
        top_k_predictors.append(sorted_x[obj][0])

    return top_k_predictors
    
percentile_k_features(x_train, y_train)



