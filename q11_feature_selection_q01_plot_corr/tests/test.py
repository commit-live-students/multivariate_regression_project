from unittest import TestCase
from ..build import plot_corr
from inspect import getargspec
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap

from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data


from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset



from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


class Test_plot_corr(TestCase):

    def test_args(self):    # Input parameters tests
    	args = getargspec(plot_corr)
    	self.assertEqual(len(args[0]), 2, "Expected arguments %d, Given %d" % (5, len(args[0])))

    def test_args_default(self):  # Input parameters defaults
        args = getargspec(plot_corr)
        self.assertEqual(args[3], (11,), "Expected default values do not match given default values")

    