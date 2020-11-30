# -*- coding: utf-8 -*-
"""

@author: roscibely

This experiment mainly uses basic Python code and the simplest data to reproduce how a
linear regression algorithm iterates and fits the existing data distribution step by step
"""


#import the necessary modules 
import numpy as np
import matplotlib.pyplot as plt 

#define data, and change list to array 

x=[3,21,22,34,54,34,55,67,89,99]
x=np.array(x)

y=[1,10,14,34,44,36,22,67,79,90]
y=np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

'''
Define related functions 
'''

#regression model ax+b
def model(a,b,x):
    return a*x+b

#loss function 
def loss_function(a,b,x,y):
    num = len(x)
    prediction = model(a,b,x)
    l=(0.5/num)*(np.square(prediction-y)).sum() 
    return l

#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize(a,b,x,y):
    num = len(x)
    prediction = model(a,b,x)
    #Update the values of A and B by finding the partial derivatives of the loss function on a and b
    da = (1.0/num) * ((prediction -y)*x).sum()
    db = (1.0/num) * ((prediction -y).sum())
    a = a - Lr*da
    b = b - Lr*db
    return a, b

#iterated function, return a and b 
def iterate(a,b,x,y,times):
    for i in range(times):
        a,b=optimize(a,b,x,y)
    return a,b

'''
Start the iteration
'''
#initialize parameters and display
a=np.random.rand(1)
b=np.random.rand(1)
Lr=1e-4 # learning rate

#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a,b = iterate(a,b,x,y,100)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
