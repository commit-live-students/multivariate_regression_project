

# Encoding of Nominal and String attribues

Check for the attributes that indicate categorical values. Also you will find atrributes that are expressed as string or characters.

## Write a function `ohe_encode()`:
use 'from sklearn.preprocessing import OneHotEncoder'.
send indices with categorical values as parameter.
Always remember, you should fit_transform on train labels and transform with same model for test labels.
The reason why I asked you to shuffle data in q01 is that you should get all labels in train set and randomise is best way to do it.

category_index - indices of categorical features.This is for avoiding one hot encoding of integer features.eg: i want to one-hot encode school but not absences.
### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X | DataFrame | compulsory |  | Train Dataframe to encode |
| X_test | DataFrame | compulsory |  | Test Dataframe to encode |
| category_index | Array | compulsory |  | array with categorical indices |
### Returns:
| Return | dtype | description |
| --- | --- | --- |
| X_transform| DataFrame | Dataframe containing feature variables encoded train |
| X_test_transform | DataFrame | Dataframe containing feature variables encoded test |
