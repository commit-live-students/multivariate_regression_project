# %load q03_ohe_encoder/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']



# def ohe_encode(df,x_test,category_index):
# ind=[]
# for i in range(len(list(df))):
#     ind.append(i)

# a=[]
# for i,j in zip(list(df),ind):
#     a.append((j,i))

# main=[]    
# for i in a:
#     for j in category_index:
#         if i[0]==j:
#             main.append(i[1])

def ohe_encode(X_train,X_test,category_index=category_index):
    X_train,X_test=label_encode(X_train,X_test)
#     X_train = pd.get_dummies(X_train)
#     X_test = pd.get_dummies(X_test)
    ohe = OneHotEncoder(categorical_features=category_index,sparse=False)
    ohe.fit(X_train)
    ohe.fit(X_test)
    X_train = ohe.transform(X_train)
    X_test = ohe.transform(X_test)            
    
    return pd.DataFrame(X_train),pd.DataFrame(X_test)  
#     nc = OneHotEncoder(categorical_features=[len(main)])
#     model1 = enc.fit_transform(x_train)
#     model2 = enc.fit_transform(x_test)
#     return model1


c=ohe_encode(x_train,x_test,category_index)
c


