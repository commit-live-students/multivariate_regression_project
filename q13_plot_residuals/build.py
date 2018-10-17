# %load q13_plot_residuals/build.py


import matplotlib.pyplot as plt

# Write your code below
def plot_residuals(y_test,y_pred,name):
    residuals = y_test - y_pred
    plt.figure(figsize=(15,8))
    plt.scatter(y_test, residuals)
    plt.title('Residual plot')
    plt.xlabel('Grade')
    plt.ylabel('Errors')
    plt.savefig(name)
    plt.show()


