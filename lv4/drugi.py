from sklearn . model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn . preprocessing import OneHotEncoder
from sklearn . metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import sklearn.linear_model as lm
import sklearn

data=pd.read_csv('data_C02_emission.csv')
data = data.drop(["Make","Model"], axis=1)
input_variables=['Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Engine Size (L)','Cylinders','Fuel Type']
output_variables=['CO2 Emissions (g/km)']

ohe = OneHotEncoder ()
X_encoded = ohe . fit_transform (data[[' Fuel Type ']]).toarray ()
data[['Fuel Type']]= X_encoded
X=data[input_variables].to_numpy()
y=data[output_variables].to_numpy()

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=1)
linearModel =lm.LinearRegression ()
linearModel.fit(X_train ,y_train)

y_test_p = linearModel.predict( X_test )
print(sklearn.metrics.max_error(y_test, y_test_p))