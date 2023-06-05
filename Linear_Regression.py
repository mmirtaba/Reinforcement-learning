import csv
import matplotlib.pyplot as plt
import numpy as np

# open the data csv file
with open('data_1d.csv',mode = 'r') as csv_file:
    data = csv.reader(csv_file,delimiter = ',')
    my_linear_data = [row for row in data]

# separate x and y data
x = [0 for i in range(100)]
y = [0 for i in range(100)]
i = 0
for row in my_linear_data:
    x[i] = float(row[0])
    y[i] = float(row[1])
    i = i+1
print(f' The X data is: {x}\n')
print(f' The Y data is: {y}\n')

# plot raw data
plt.plot(x,y,'ko')
plt.title('Raw data')
plt.xlabel('X axis', fontsize = 12)
plt.ylabel('Y axis', fontsize = 12)


x = [[i] for i in x]
y = [[i] for i in y]
x = np.array(x)
y = np.array(y)

# Gradient_Descent algorithm
learning_rate = 0.0001
a = 0
b = 0

def Gradient_Descent(x,y,a,b,learning_rate):
    j_a = 0
    j_b = 0
    N = x.shape[0]
    for xi,yi in zip(x,y):
        j_a += -2*xi*(yi-(a*xi+b))
        j_b += -2*(yi-(a*xi+b))

    a = a - learning_rate*j_a*(1/N)
    b = b - learning_rate*j_b*(1/N)

    return a,b


for i in range(50):
    a,b = Gradient_Descent(x,y,a,b,learning_rate)

print(f'Estimated slope is: {a}, and estimated intercept is: {b}')

x_min = min(x)
x_max = max(x)

y_min = a*x_min + b
y_max = a*x_max + b
plt.plot([x_min, x_max],[y_min, y_max])
plt.show()
