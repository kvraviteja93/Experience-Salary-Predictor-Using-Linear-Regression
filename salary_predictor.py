#Simple Linear Regression
#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing The Dataset
dataset=pd.read_csv("Salary_Data.csv")

#Separating Independent and Dependent Variables
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Splitting Dataset Into Training and Test Set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#Creating The Linear Regression Model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
#The Machine Learning Model Has Now Learned

#Now that the model is created it's time to predict the results
y_predictions=lr.predict(x_test)
#The Predictions Are Done!
#Now, predicting any new data point
'''For example, I need to check what would 
be the salary of an employee having an 
experience of 2.5 years.'''
lr.predict(2.5)

#Visualizing The Predictions(Training Set)
%matplotlib inline
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lr.predict(x_train))
plt.title('Salary Vs. Experience')
plt.xlabel('Experience(In Years)')
plt.ylabel('Salaries')
plt.show()

#Visualizing The Predictions(Test Set)
%matplotlib inline
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,lr.predict(x_train))
plt.title('Salary Vs. Experience')
plt.xlabel('Experience(In Years)')
plt.ylabel('Salaries')
plt.show()



