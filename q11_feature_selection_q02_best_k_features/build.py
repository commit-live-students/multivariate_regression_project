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
# Write your code below

def percentile_k_features(x_train, y_train, k=50):
    selector = SelectPercentile(f_regression,percentile=k)
    selector.fit_transform(x_train, y_train)
    scores = selector.scores_[selector.get_support()]
    features = x_train.columns.values[selector.get_support()]
    features_scores_list = list(zip(features,scores))
    df = pd.DataFrame(features_scores_list, columns=['Features','Scores'])
    sorted_list = df.sort_values('Scores',ascending=False)
    top_k_predictors = list(sorted_list['Features'])
    return top_k_predictors

# percentile_k_features(x_train, y_train, k=50)


