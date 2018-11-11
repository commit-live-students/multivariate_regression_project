# %load q04_data_visualisation/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q04_data_visualisation.build import visualise_data
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
data = load_data('data/student-mat.csv') 
x_train, x_test, y_train, y_test =  split_dataset(data)
x_train,x_test = label_encode(x_train,x_test)
df_train = x_train.join(y_train)
def visualize_data(df_train, path):
    plot = scatter_matrix(df_train)
    plt.show();
    return plot




