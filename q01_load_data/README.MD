# Let's get going - Load Data

We will start with loading the data into a variable, which we will use for further analysis

## write a function `load_data()` that 

* Loads a dataset using panda's read_table function with the following specifications:

**Notice the separator of your datafile is not comma but a semiciolon.**

It is always important to check the separator before starting.
read_csv will help you read a comma separated file but any other delimiter has to be handled by read_table with delimiter given as "sep=';'"
Also, dont forget to shuffle your data before you start.
use random seed 7 if you like.
### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| path | string | compulsory |  | path to the file csv |


### Returns:

| Return | dtype | description |
| --- | --- | --- |
| df | Pandas DataFrame | DataFrame for creating our model |