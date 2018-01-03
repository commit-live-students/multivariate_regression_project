

# Encoding of Nominal and String attribues

Check for the attributes that indicate nominal values. Also you will find atrributes that are expressed as string or characters.

## Write a function `label_encode()`:
use 'from sklearn.preprocessing import LabelEncoder' import statement to check the first instance (row) for all attrributes. if you find anything non-numeric, assume feature to be non-numeric and pass it through label encoder to obtain its encoding.


### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X | DataFrame | compulsory |  | Train Dataframe to encode |
| X_test | DataFrame | compulsory |  | Test Dataframe to encode |


### Returns:
| Return | dtype | description |
| --- | --- | --- |
| X_transform| DataFrame | Dataframe containing feature variables encoded train |
| X_test_transform | DataFrame | Dataframe containing feature variables encoded test |
