# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for the marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Santhosh L
RegisterNumber: 212222100046
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("/content/student_scores (1).csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```
### head
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("/content/student_scores (1).csv")
df.head()
```
### tail
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("/content/student_scores (1).csv")
df.tail()
```
## Output:
![image](https://github.com/Dhiyanesh24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118362288/3b5dc2bb-3e07-48fe-b32c-4a896a3207c5)
![image](https://github.com/Dhiyanesh24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118362288/15ed20a1-6b20-4fcd-bdfb-f146659e5dce)
![image](https://github.com/Dhiyanesh24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118362288/4f496d86-ddd0-4675-ac54-4e88a0443b97)
### head
![image](https://github.com/Dhiyanesh24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118362288/dbdc5e69-aa5a-4193-bb7a-819db32c425e)
### tail
![image](https://github.com/Dhiyanesh24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118362288/02c3be53-af70-4c04-84e3-9937b87f4d86)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
