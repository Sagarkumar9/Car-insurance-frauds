import os
os.getcwd()
os.chdir("C:/Users/namra/Desktop/6th SEM/PROJECT")
import pandas as pd
D1=pd.read_csv("laptop_data.csv")
D1.dtypes
D1.isnull().sum()

fd=D1
fd = pd.DataFrame(fd)
fd["Inches"] = fd["Inches"].fillna(method='ffill')
fd["Ram"] = fd["Ram"].fillna(method='ffill')


fd.isnull().sum()

fd.Company.unique()
fd.TypeName.unique()
fd.Inches.unique()
fd.ScreenResolution.unique()
fd.Cpu.unique()
fd.Ram.unique()
fd.Memory.unique()
fd.Gpu.unique()
fd.OpSys.unique()
fd.Weight.unique()



import pingouin as pp

pp.welch_anova(fd,dv="Price",between="Company")
pp.welch_anova(fd,dv="Price",between="TypeName")      
pp.welch_anova(fd,dv="Price",between="Inches") 
pp.welch_anova(fd,dv="Price",between="ScreenResolution")    
pp.welch_anova(fd,dv="Price",between="Cpu")    
pp.welch_anova(fd,dv="Price",between="Ram")
pp.welch_anova(fd,dv="Price",between="Memory")
pp.welch_anova(fd,dv="Price",between="Gpu")
pp.welch_anova(fd,dv="Price",between="OpSys")
pp.welch_anova(fd,dv="Price",between="Weight")

fd = fd.drop('Unnamed: 0', axis=1)

from sklearn.model_selection import train_test_split
x = fd.drop('Price', axis=1)
y = fd['Price']
x=pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


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
 