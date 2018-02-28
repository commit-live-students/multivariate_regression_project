# %load q11_feature_selection_q01_plot_corr/build.py
#import sys
#sys.setrecursionlimit(100000)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data


from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset



from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


# =============================================================================
# To visualise data, you need to pass training data only as the assumption holds that test set is unknown data and obviously,you cant not make decision based on unseen data :-p

#Remember to concatenate training features and labels if you want to check that scatterplots which I would prefer.You are free to explore labels to labels, features to features ,etc scatterplots as you want by passing arguments
#============================================================================
#visualise_data(pd.concat([x_train,y_train],axis=1),"../images/data_image.png")

# Write your solution here:
def plot_corr(data, size=11):
    plt.figure(figsize=(12,8))
    sns.heatmap(data.corr(), cmap='YlOrRd')
    return plt.show()

#plot_corr1(df,11)
