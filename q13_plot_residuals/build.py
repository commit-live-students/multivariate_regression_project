# %load q13_plot_residuals/build.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Write your code below
def plot_residuals(y_test, y_pred, name):
    plt.plot(y_test, y_pred)
    plt.show()


