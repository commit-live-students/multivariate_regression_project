from sklearn.preprocessing import OneHotEncoder
import pandas as pd
# %load q03_ohe_encoder/build.py
#categorical_variables = [x for x in range(len(list(X))) if X[X.columns[x]].dtype=='object']
def ohe_encode(X,X_test,category_index=[0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22]):
    X_transform = X.copy()
    X_test_transform = X_test.copy()
    ohe = OneHotEncoder(categorical_features=category_index)
    X_transform = ohe.fit_transform(X_transform)
    X_test_transform = ohe.fit_transform(X_test_transform)
    return X_transform,X_test_transform


