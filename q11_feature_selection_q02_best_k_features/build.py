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
def percentile_k_features(X,y, k=50):
    sp = SelectPercentile(f_regression,percentile=k)
    sp.fit_transform(X,y)
    features = X.columns.values[sp.get_support()]
    scores = sp.scores_[sp.get_support()]
    fs_score = list(zip(features,scores))
    df = pd.DataFrame(fs_score,columns=['Name','Score'])
    return df.sort_values(['Score','Name'],ascending = [False,True])['Name'].tolist()
