# %load q13_plot_residuals/build.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Write your code below
def plot_residuals(y_test,y_pred,name):
    residual = y_test - y_pred
    plt.figure(figsize=(16,7))
    plt.scatter(y_test, residuals)
    plt.title('Residual plot')
    plt.xlabel('Grade')
    plt.ylabel('Residuals')
    plt.show()
    

