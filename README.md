# Guided Project - Multivariate regression
Hello Data Scientists,
Lets try our out guided project of predicting students performance based on given features.
This is type of multivariate multioutput regression problem and you will be getting chance to showcase your learning experiences gathered so far.
We will be presenting multiple approaches to solve the given problem in the real-life way so consider this as simulation to actual data scientists problems available out there.
Please feel free to try out other approaches as well if you deem fit, the show cased here will just be the approach.
My target here will be  to demonstrate general approach to a problem.
## What we have learnt so far..

- Linear regression
- Feature Selection
- Feature Engineering
- Advanced linear regression techniques

## Dataset
Predict student performance in secondary education (high school).


#### Features:

# Attributes for both student-mat.csv (Math course):
1. school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira) 
2. sex - student's sex (binary: 'F' - female or 'M' - male) 
3. age - student's age (numeric: from 15 to 22) 
4. address - student's home address type (binary: 'U' - urban or 'R' - rural) 
5. famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3) 
6. Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart) 
7. Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education) 
8. Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education) 
9. Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
10. Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
11. reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other') 
12. guardian - student's guardian (nominal: 'mother', 'father' or 'other') 
13. traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour) 
14. studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours) 
15. failures - number of past class failures (numeric: n if 1<=n<3, else 4) 
16. schoolsup - extra educational support (binary: yes or no) 
17. famsup - family educational support (binary: yes or no) 
18. paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no) 
19. activities - extra-curricular activities (binary: yes or no) 
20. nursery - attended nursery school (binary: yes or no) 
21. higher - wants to take higher education (binary: yes or no) 
22. internet - Internet access at home (binary: yes or no) 
23. romantic - with a romantic relationship (binary: yes or no) 
24. famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent) 
25. freetime - free time after school (numeric: from 1 - very low to 5 - very high) 
26. goout - going out with friends (numeric: from 1 - very low to 5 - very high) 
27. Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high) 
28. Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high) 
29. health - current health status (numeric: from 1 - very bad to 5 - very good) 
30. absences - number of school absences (numeric: from 0 to 93) 

# These grades are related with the course subject, Math: 
31. G1 - first period grade (numeric: from 0 to 20) 
31. G2 - second period grade (numeric: from 0 to 20) 
32. G3 - final grade (numeric: from 0 to 20, output target)

## What you will learn solving this ?

- Learn systematic approach to select features
- Compare various regression techniques
- Emphasis will be given on correlations between dependant features to take call on approaches
- Also try out advanced regressions to check what works best for dataset. 


### General Notes to approach problems are:
-How to approach a ML problem
    1.import data
    2.missing data
        a.remove the missing lines - dangerous
        b.imputation - take mean of column - sklearn.preprocessing.Imputer

    3. convert categorical data	
    4.splitting datasets - 
    5.Feature Scaling
        a. Standardisation - (x-mean(x))/std_dev(x) 
        b.Normalisation		 - (x-min(x))/(max(x)-min(x))
    6.Apply classifier and test on split
    7.Draw conclusions by plottig if required	

Seems like you are all fired up to put a test to your knowledge.

Let's get started!
