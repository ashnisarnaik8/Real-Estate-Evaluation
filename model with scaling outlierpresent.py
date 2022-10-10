# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:00:00 2022

@author: HP
"""

import numpy as np
import pandas as pd

df = pd.read_csv(r"D:\Real estate valuation data set (1).csv")
df
df = df.rename(columns = {'X2 house age': 'Age',
                          'X3 distance to the nearest MRT station': 'nearest_mrt',
                          'X4 number of convenience stores': 'no_stores',
                          'Y house price of unit area': 'Value'})



dum_df = pd.get_dummies(df, drop_first=True)
#area0 is compressed
dum_df.columns
X = dum_df.drop(columns = 'Value')
X
y = dum_df['Value']
y
#we have sepreated dependent and independent var where x is independent and y is dependent

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[['Age', 'nearest_mrt',
   'no_stores']] = scaler.fit_transform(X[['Age', 'nearest_mrt',
                                         'no_stores']])
   
from sklearn.model_selection import train_test_split 
#as in our question we have give test data=1/3 and train data=2/3
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=123)

#################################################################################################################
#we can use various models and chose the best one 
##################################LINEAR REGRESSION##############################################################

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred
#we get predicted value at each data point

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y_test, y_pred)))
#it is used to see how much a model is closed to the actual data lower the value better
print(mean_absolute_error(y_test, y_pred))
#is difference between forecast value and actual value lower the value better
print(r2_score(y_test, y_pred))
#rsq tells us how well a model predicts thevalue of response variable here in terems of percentage here 69% 


################################OLS###########################################################

import statsmodels.api as sm
import pandas as pd
 
# reading data from the csv
df
 
# defining the variables
X_train
y_train
 
# adding the constant term
X_train = sm.add_constant(X_train)
X_train 
# performing the regression
# and fitting the model
result = sm.OLS(y_train, X_train).fit()
result 
# printing the summary table
print(result.summary())
####################################################################################################