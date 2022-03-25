
import re
from turtle import color
from cv2 import IMWRITE_PAM_FORMAT_GRAYSCALE, mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
''' đọc dữ liệu từ file'''
df = pd.read_csv('Mall.csv')
df.info()
# print(df)

x = df.iloc[:,[3,4]].values
# print(x)

nk = []
''' tìm k trong khoảng 2-11 cụm'''
for i in range(2,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, random_state=0).fit(x)
    nk.append(kmeans.inertia_)
plt.plot(range(2,11),nk,marker = 'o')
# print(nk)
plt.show()

''' tìm được k= 5 '''
y_means = kmeans.fit_predict(x)
X0 = x[y_means == 0, :]
X1 = x[y_means == 1, :]
X2 = x[y_means == 2, :]
X3 = x[y_means == 3, :]
X4 = x[y_means == 4, :]
'''gán nhãn cho từng cụm'''
def kmeans_display(X, label):
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 10)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 10)
    plt.plot(X2[:, 0], X2[:, 1], 'o', markersize = 10)
    plt.plot(X3[:, 0], X3[:, 1], 'o', markersize = 10)
    plt.plot(X4[:, 0], X4[:, 1], '*', markersize = 10)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(x,y_means)


# plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,marker='o',color='red')
# plt.scatter(x[y_means==1,0],x[y_means==1,1],s=200,marker='o',color='blue')
# plt.scatter(x[y_means==2,0],x[y_means==2,1],s=250,marker='o',color='green')
# plt.scatter(x[y_means==3,0],x[y_means==3,1],s=150,marker='o',color='yellow')
# plt.scatter(x[y_means==4,0],x[y_means==4,1],s=300,marker='o',color='cyan')


''' tìm tọa độ center'''
k = []
def findCenter(x):
    z = []
    z.append(x[:,0].mean())
    z.append(x[:,1].mean())
    k.append(z)
findCenter(X0)
findCenter(X1)
findCenter(X2)
findCenter(X3)
findCenter(X4)
print(k)
plt.scatter(k[:0],k[:1])
plt.show()

