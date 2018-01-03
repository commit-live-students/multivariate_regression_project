

# Analysis of features to check for missing values

Now that you have imported the data, lets take look at this data.
for this tak you should use pandas' describe function and check out what comes on your way.
Purpose of this assignment is to understand how to take call on feature engineering and selection.
This is very important part of any Machine Learning project.The complication is not in code but in the observation and analysis of this siomple function's results.

## Write a function `describe_df()` which prints the statistics of your features:
### Analysis expectations: check if you can tell if missing values imputation is necessary.If not then why? Also are their any properties or conclusions you can draw from the describe functions and what are they?
Observe feature.describe given. For analysing missing value, check min row of each feature and try resoning if it is valid to have 0 as value. In our case, Medu,FEdu and absences has value 0. MEdu and FEdu are nominal values with 0 as valid encoding for none. So it is actual value and not missing value.Also , intuitively in real life problem, it is safe to assume in this case for attendance to be 100% hence absence=0 is valid value.
In the datasets downloaded over web, make sure you read description of dataset(student.txt in our case) or the analysis on website.
missing value analysis is not applicable as we can read in data description downloaded - student.txt that many features do take value to be 0.




You may notice that we have skipped outlier detection here due to all being nominal value chances of outliers is small.
In case of categorical data, outliers are calculated on the basis of frequency of particular value of columns.
I suggest you inspect our data with x_train.apply(pd.value_counts)
You will see great deal of NaN except absences column.
This means the values have particular range as in many cases its upto 5 (categorical data converted to nominal data so obvious.)
Now as relying on absences only entirely wont be fair in this case given limited data.
But you are free to experiment.
### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| df | DataFrame | compulsory |  | Input Dataframe to analyse |


### Returns:
Not applicable as printed on console.
