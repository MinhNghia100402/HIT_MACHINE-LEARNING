

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from matplotlib.colors import ListedColormap
df = datasets.load_iris()
#  dùng chiều dài cánh hoa và chiều rộng cánh hoa để phân biệt
x= df['data'][:,[2,3]]
# lấy label
y= df['target'][:]



X_train,X_test,Y_train,Y_test= train_test_split(x,y,test_size=20,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

#  phân lớp dữ liệu 
cm = confusion_matrix(Y_test, y_pred)

X_set, y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# vẽ các đường đồng mức và các đường viền đã tô, tương ứng
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             cmap = ListedColormap(('red', 'yellow','blue')))
 # vẽ các điểm dữ liệu   
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('yellow', 'green','red'))(i), label = j)

                
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()
