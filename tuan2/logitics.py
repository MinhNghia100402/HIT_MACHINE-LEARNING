import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
df = pd.read_csv('data_logistic.csv')
df.info()
''' change data becoming interger'''
df['Gender'] = df['Gender'].map({'Male' : 1,'Female' : 0})
''' create data'''
a = df.iloc[:,0:4]
b = df.iloc[:,4:5]

''' apply train_test_split'''

x_train , x_test, y_train, y_test = train_test_split(a,b,test_size=0.5,random_state=0)

''' training model'''

Logistic = LogisticRegression()
Logistic.fit(x_train,y_train)
y_pred = Logistic.predict(x_test)

'''Evaluate model'''

from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

