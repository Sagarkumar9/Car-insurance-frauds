import os
os.getcwd()
os.chdir("C:/Users/namra/Desktop/6th SEM/PROJECT")
os.getcwd()
import pandas as pd
import numpy as np
fd = pd.read_csv("hadoop.csv")
fd.isnull().sum()

fd["Invoice ID"].unique()
fd["Branch"].unique()
fd["City"].unique()
fd["Customer type"].unique()
fd["Gender"].unique()
fd["Product line"].unique()
fd["Unit price"].unique()
fd["Quantity"].unique()
fd["Tax 5%"].unique()
fd["Total"].unique()
fd["Date"].unique()
fd["Time"].unique()
fd["Payment"].unique()
fd["cogs"].unique()
fd["gross margin percentage"].unique()
fd["gross income"].unique()
fd["Rating"].unique()

fd = pd.DataFrame(fd)
fd['Invoice ID'] = fd['Invoice ID'].str.replace('-', '')



fd["Branch"] = fd["Branch"].fillna(method='ffill').fillna(method='bfill')
fd["City"] = fd["City"].fillna(method='ffill').fillna(method='bfill')
fd["Customer type"] = fd["Customer type"].fillna(method='ffill').fillna(method='bfill')
fd["Gender"] = fd["Gender"].fillna(method='ffill').fillna(method='bfill')
fd["Product line"] = fd["Product line"].fillna(method='ffill').fillna(method='bfill')
fd["Payment"] = fd["Payment"].fillna(method='ffill').fillna(method='bfill')
fd["gross margin percentage"] = fd["gross margin percentage"].fillna(method='ffill').fillna(method='bfill')
fd["cogs"].fillna(fd["cogs"].mean(), inplace=True)
fd["gross income"].fillna(fd["gross income"].mean(), inplace=True)

fd = fd.drop(["Invoice ID","Date","Time"], axis=1)


from sklearn.model_selection import train_test_split
x = fd.drop('Total', axis=1)
y = fd['Total']
x=pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)                         

import xgboost
from xgboost import XGBRFRegressor
model=XGBRFRegressor()
model.fit(x_train,y_train)
y2_pred=model.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y2_pred)                        

from sklearn.tree import DecisionTreeRegressor
m=DecisionTreeRegressor()
m.fit(x_train,y_train)
y3_pred=m.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test, y3_pred)                        

from sklearn.ensemble import RandomForestRegressor
m1=RandomForestRegressor()
m1.fit(x_train,y_train)
y4_pred=m1.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test, y4_pred)
