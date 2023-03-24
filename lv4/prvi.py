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

data=pd.read_csv('data_C02_emission.csv')
data = data.drop(["Make","Model"], axis=1)
input_variables=['Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Engine Size (L)','Cylinders']
output_variables=['CO2 Emissions (g/km)']
X=data[input_variables].to_numpy()
y=data[output_variables].to_numpy()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
plt.scatter(X_train[:,0],y_train,c="blue")
plt.scatter(X_test[:,0],y_test,c="red")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

plt.scatter(X_train[:,1],y_train,c="blue")
plt.scatter(X_test[:,1],y_test,c="red")
plt.xlabel("Fuel Consumption Hwy (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

plt.scatter(X_train[:,2],y_train,c="blue")
plt.scatter(X_test[:,2],y_test,c="red")
plt.xlabel("Fuel Consumption Comb (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

plt.scatter(X_train[:,3],y_train,c="blue")
plt.scatter(X_test[:,3],y_test,c="red")
plt.xlabel("Fuel Consumption Comb (mpg)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

plt.scatter(X_train[:,4],y_train,c="blue")
plt.scatter(X_test[:,4],y_test,c="red")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

plt.scatter(X_train[:,5],y_train,c="blue")
plt.scatter(X_test[:,5],y_test,c="red")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

sc = StandardScaler()
X_train_n = sc.fit_transform ( X_train )
X_test_n = sc.transform ( X_test )

plt.subplot(2, 1, 1)
plt.hist(X_train[:, 1])
plt.title('Before Scaling')
plt.xlabel('Fuel Consumption Hwy (L/100km)')
plt.ylabel('Frequency')

plt.subplot(2, 1, 2)
plt.hist(X_train_n[:, 1])
plt.title('After Scaling')
plt.xlabel('Fuel Consumption Hwy (L/100km)')
plt.ylabel('Frequency')
plt.show()

linearModel =lm.LinearRegression ()
linearModel.fit(X_train_n ,y_train)

y_test_p = linearModel.predict( X_test_n )

plt.scatter(y_test,y_test_p,c="blue")

plt.xlabel("Fuel Consumption Hwy (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()


MAE = mean_absolute_error ( y_test , y_test_p )
MAPE=mean_absolute_percentage_error(y_test,y_test_p)
MSE=mean_squared_error(y_test,y_test_p)
RMSE=np.sqrt(mean_squared_error(y_test,y_test_p))
R=r2_score(y_test,y_test_p)
print(MAE,MAPE,MSE,RMSE,R)

