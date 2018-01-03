import sys, os
# sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from unittest import TestCase
from ..build import split_dataset
from inspect import getargspec
import pandas
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test = split_dataset(df)


class TestRead_split_dataset(TestCase):

	def test_split_dataset_args(self):
		arg = getargspec(split_dataset).args
		self.assertEqual(len(arg),1 ,"Expected argument(s) %d, Given %d" % (1,len(arg)))

	def test_x_train_return_instance_X(self):
		self.assertIsInstance(x_train, pandas.DataFrame,"The Expected return type does not match with the given return type")

	def test_x_test_return_instance_y(self):
		self.assertIsInstance(x_test, pandas.DataFrame,"The Expected return type does not match with the given return type")

	def test_y_train_return_instance_y(self):
		self.assertIsInstance(y_train, pandas.Series,"The Expected return type does not match with the given return type")

	def test_y_test_return_instance_y(self):
		self.assertIsInstance(y_test, pandas.Series,"The Expected return type does not match with the given return type")
	
	def test_x_train_shape(self):
	    self.assertEqual(x_train.shape , (316, 32), "The Expected return shape does not match with the given return shape")

	def test_x_test_shape(self):
	    self.assertEqual(x_test.shape , (79, 32), "The Expected return shape does not match with the given return shape")

	def test_y_train_shape(self):
	    self.assertEqual(y_train.shape , (316,), "The Expected return shape does not match with the given return shape")

	def test_y_test_shape(self):
	    self.assertEqual(y_test.shape , (79, ), "The Expected return shape does not match with the given return shape")
