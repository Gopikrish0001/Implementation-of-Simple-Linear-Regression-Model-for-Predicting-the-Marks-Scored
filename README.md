# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
##NAME :GOPIKRISHNAN M
##REGNO:212223043001


## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 
 
 
 

## Program:
```
/*
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
##HEAD VALUE
![Screenshot 2025-03-07 091010](https://github.com/user-attachments/assets/f69822e7-2934-4533-adfa-058565d931f0)


##TAIL VALUE
![image](https://github.com/user-attachments/assets/eceafc6f-38e0-43f3-b6e4-ccd8f437cf6d)


##COMPARE DATASET
![image](https://github.com/user-attachments/assets/ee5dbe47-1c00-4e3c-be2b-8079708f6250)

##PERDICATION VALUES OF X AND Y
![Screenshot 2025-03-07 093916](https://github.com/user-attachments/assets/5e1480b1-4037-4fff-bf16-dfa29e6941e4)

##TRAINING SET
![Screenshot 2025-03-07 091329](https://github.com/user-attachments/assets/ffcf44df-a9d9-4dd2-8fce-9773a0f3aadf)

##TESTING SET
![Screenshot 2025-03-07 091355](https://github.com/user-attachments/assets/83e35a5c-ce8e-4a0a-a4ec-6478b4882f66)


##MSE,MAE and RMSE
![Screenshot 2025-03-07 091418](https://github.com/user-attachments/assets/9916f320-6afe-4640-8966-0a74f7996cd9)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
