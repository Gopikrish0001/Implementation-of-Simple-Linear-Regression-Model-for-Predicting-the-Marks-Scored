# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## NAME:GOPIKRISHNAN M
## REGNO:212223043001


## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas. 
~~~ 

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
*/
```

## Output:

## HEAD VALUES
![WhatsApp Image 2025-03-07 at 09 12 47_0427de93](https://github.com/user-attachments/assets/b3b44189-e367-4327-b252-68117d56b38a)


## TAIL VALUES
![WhatsApp Image 2025-03-07 at 09 12 47_6a42e85e](https://github.com/user-attachments/assets/073e205e-7c93-43eb-85df-a53f98184d86)


## COMPARE DATASET
![WhatsApp Image 2025-03-07 at 09 12 46_37ac983c](https://github.com/user-attachments/assets/a9d8ea93-e698-4080-ab7e-95428da426e0)


## PREDICATION OF X AND Y VALUES
![image](https://github.com/user-attachments/assets/726ceafb-2c75-4ba5-bf1f-ebae360afbc7)


## TRAINING SET

![WhatsApp Image 2025-03-07 at 09 12 46_48a80912](https://github.com/user-attachments/assets/090e2ca7-dc58-4061-8c3c-cc2cb38aab4e)

## TESTING TEST
![WhatsApp Image 2025-03-07 at 09 12 47_d27ada0c](https://github.com/user-attachments/assets/df8a62fc-f3f7-42e6-865d-f0925d25af3d)



## MSE,MAE AND RMSE
![Screenshot 2025-03-07 091418](https://github.com/user-attachments/assets/65595a81-817f-452d-bb7d-5b5d7c388cc1)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
