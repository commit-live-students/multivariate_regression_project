# %load q13_plot_residuals/build.py


import matplotlib.pyplot as plt
# Write your code below
def plot_residuals(y_test,y_pred,name):
    plt.scatter(y_test,y_pred)
    plt.title('name')
    plt.show()


