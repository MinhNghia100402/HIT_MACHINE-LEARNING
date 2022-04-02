
from itertools import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import math


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df = pd.read_csv('Network_Ads.csv')

# df.info()
''' xuất hiện missing values '''

df['EstimatedSalary'] = imputer.fit_transform(df['EstimatedSalary'].values.reshape(-1,1))[:,0]

''' xử lí missing values cho tất cả giá trị nan bằng giá trị trung bình  mean '''
# df.info()
''' dùng cột age và EstimatedSalary để làm điểm dữ liệu đầu vào 
    dùng cột Purchased để dùng làm thuộc tính cho điểm dữ liệu đó
    từ đó có thể quyết định được điểm đầu vào thuộc class nào 0 hoặc 1
    bài toán giải quyết vấn để như : có thể quyết định mua gì đó của một người tùy vào số tuổi và thu nhập của họ 
'''

''' đưa vào một điểm dữ liệu tương đương với số tuổi và mức lương của họ để dự đoán xem là họ có thể đủ tiền mua 
    hay không + đủ tiền : thuộc lớp 1
              + không đủ tiền : thuộc lớp 0
'''
a = df.iloc[:,[1,2,3]].values
b = df.iloc[:,[3]].values
X_train ,X_test,Y_train,Y_test = train_test_split(a,b,test_size=50)


paint = []
for i in Y_train:
    if(i==1):
        paint.append('green')
    else:
        paint.append('blue')

'''point check'''



person = [50,35000]

plt.scatter(X_train[:,0],X_train[:,1],c=paint)
plt.scatter(person[0],person[1],color='red',edgecolors="k")
plt.xlabel('Age',size=20)
plt.ylabel('Salary',size=20)



def distance(pointA,pointB):
    return math.sqrt((pointB[0]-pointA[0])**2 + (pointB[1] - pointA[1])**2)



def kNearestNeighbor(trainSet, point,k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[2],
            "value": distance(item, point)
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    if k%2==0: k+=1
    return labels[:k]
d = kNearestNeighbor(X_train,person,5)
print(d)
max ,c,f= 0,0,0
for i in set(d):
    f+=d.count(i)
    if(d.count(i)>max):
        max = d.count(i)
        c= i
print("độ chính xác là : ",float((max/f)*100))
print('điểm đó thuộc lớp : ',c)
plt.show()


