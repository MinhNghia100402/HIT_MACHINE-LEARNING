import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data_linear.csv')
x = np.array(df['Diện tích'])

y = np.array(df['Giá']) 

''' biến đổi x thành ma trậ chuyển vị '''


xg = x.T


# slope, intercept, r, p, std_err = stats.linregress(x, y)

# def myfunc(x):
#   return slope * x + intercept

# mymodel = list(map(myfunc, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()

''' tính theo công thức cho sẵn 
    y = a+bx
    b = [tổng sigma(1-n) (xi*yi - n*x.mean()*y.mean())] / tổng sigma(1-n) (xi**2 - n*(x.mean()**2))
    a = y.mean() - b*x.mean()

'''
def cost(x,y):
    j = 0
    j= sum(x*y-np.size(x)*x.mean()*y.mean())
    return j
def loss(x):
    z = 0
    z =  sum(x**2-np.size(x)*(x.mean())**2)
    return z
c = cost(x,y)
d= loss(x)
b = c/d
a = y.mean()-b*x.mean()
def data(e):
    return a+b*e
yg = []
def myMD():
    for i in x:
        yg.append(data(i))
    return yg
toss = myMD()
plt.scatter(x,y)
plt.plot(x,toss)
plt.show()



