import sys, os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from unittest import TestCase
from ..build import load_data
from inspect import getargspec
import pandas

path = 'data/student-mat.csv'
df = load_data(path)

class TestRead_csv_data_to_df(TestCase):

	def test_read_csv_data_to_df_args(self):
		arg = getargspec(load_data).args
		self.assertEqual(len(arg),1 ,"Expected argument(s) %d, Given %d" % (1,len(arg)) )

	def test_read_csv_data_to_df_return_instance(self):
		self.assertIsInstance(df, pandas.DataFrame,"The Expected return type does not match with the given return type")

	def test_read_csv_data_to_df_return_shape(self):
	    self.assertEqual(df.shape , (395, 33), "The Expected return shape does not match with the given return shape")
