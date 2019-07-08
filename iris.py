import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import datasets
iris=datasets.load_iris()

iris_data=iris.data
iris_data=pd.DataFrame(iris_data , columns=iris.feature_names)
iris_data['class']=iris.target
iris_data.head

#understanding data
iris.target_names
print(iris_data.shape)
p=iris_data.describe()

#train test split
from sklearn.model_selection import train_test_split
X=iris_data.values[:,0:4]
Y=iris_data.values[:,4]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


#applying Different Models
#logistic Regression
model=LogisticRegression()
model.fit(x_train,y_train)
predictionsL=model.predict(x_test)
print(accuracy_score(y_test,predictionsL))

#Random Forest
modelR=RandomForestClassifier(n_estimators=500 , random_state=0)
modelR.fit(x_train,y_train)
predictionsR=modelR.predict(x_test)
print(accuracy_score(y_test,predictionsR))

#KNN
modelK=KNeighborsClassifier()
modelK.fit(x_train,y_train)
predictionsK=modelK.predict(x_test)
print(accuracy_score(y_test,predictionsK))

#SVM
modelS=SVC()
modelS.fit(x_train,y_train)
predictionsS=modelS.predict(x_test)
print(accuracy_score(y_test,predictionsS))
