import math
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
''' hàm cost trả về giá trị của hàm số ở đây em lấy ví dụ hàm số y = x^2 +4x'''
def cost(x):
    return x**2 + 4*math.sin(x)
''' hàm grad trả về giá trị đạo hàm của hàm số'''
def grad(x):
    return 2*x + 4*math.cos(x)

''' hàm myGD để thực hiện thuật toán gradient descent '''
def myGD(x0,learningrate):
    '''tạo một list chứa các giá trị x tìm được sau mỗi vòng lặp'''
    x= [x0]
    '''số vòng lặp tùy chọn'''
    for i in range(100):
        # áp dụng công thức toán học của gradient descent cho hàm một biến
        x_new = x[-1] - learningrate*(grad(x[-1]))
        # nếu giá trị của x < giá trị 1e-3 thì kết thúc vòng lặp
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return(x,i)
(a,vt )= myGD(5,0.1)
print('các giá trị của x tìm được là ',a,'\n sau ',vt,' vòng lặp')
axis = np.array(a)
y = []
for i in axis:
    y.append(grad(i))
plt.plot(axis,y,marker='o',color='red')
plt.title('THUẬT TOÁN GRADIENT DESCENT')
plt.xlabel('x',color='red')
plt.ylabel('y',color='red')
plt.show()
