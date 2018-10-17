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
def percentile_k_features(x_train,y_train,k=50):
    kbest = SelectPercentile(f_regression,k)
    kbestf = kbest.fit(x_train,y_train)
    ans = []
    c = list(x_train.columns)
    scores = list(kbestf.scores_)
    temp = scores
    d = {}
    for i in range(0,len(c)):
        d[c[i]] = scores[i]

    temp.sort(reverse=True)
    for i in range(0,16):
        for val in d.keys():
            if d[val] == temp[i]:
                ans.append(val)

    return ans

#percentile_k_features(x_train,y_train,k=50)
