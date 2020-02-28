import numpy as np
from numpy import *
from math import*

import matplotlib.pyplot as plt
import sys

M = 1
leng = 40
h = 0.5
EPS_MIN = 1e-6
EPS_MAX = 1e-4
h_min   = 0.009
h_max   = 2
n = int(leng/h)
#print("n=", n)
y_k = np.zeros(M)
y = np.zeros(M)
f = np.zeros(M)
k1 = np.zeros(M)
k2 = np.zeros(M)
k3 = np.zeros(M)
k4 = np.zeros(M)
f = np.zeros(M)
y_ = np.zeros(M)
y_new = np.zeros(M)
y_new = np.zeros(M)
y_k1 = np.zeros(M)
xx = open("x.txt", "w")
my = open("runge.txt", "w")
my_gr = open("my_gr.txt", "w")

def F(f, y_k, x) :

    f[0] = x**3 + 1#(-1.0) * x * exp((x**2)/(-2.0))x**3 + 1
    #f[1] = cos(x)#x**2 + x + 1
    return f
def y_true(x):
    #y =  exp((x**2)/(-2.0))
    y = 0.25 * (x**4) + x + 1
    #y[1] = sin(x)# (x**3)/3 + 0.5 * (x**2) + x + 2
    return y
   
#def norma(y1,y2):
#    s = np.dot(y1-y2, y1-y2)  #np.dot - скалярное произведение
#    global n
#    n = sqrt(s)
#    return n
def norma(y_k1_h, y_k1):
    summ = 0
    for i in range (0, M):
        summ += pow(y_k1_h[i]-y_k1[i],2)
    global n
    n = sqrt(summ)
def out(x, h, y1, y2,y3): #y1 - из рунге, y2 - из рунге с мельчением, y3-настоящее
    norm=norma_diff(y2,y3)    #между наст и мельчением
    norm1=norma_diff(y1,y2) #между двумя рунге

def runge (x, h):
    global F

    global y_k, y_tmp, y_k1
    global k1, k2, k3, k4, y_
    
    F(k1, y_k, x)   #считаем к1

    F(k2, y_k + k1*0.5*h, x + 0.5 * h) #считаем к2
   
    F(k3,  y_k + 0.5 * k2 * h, x + 0.5 * h) #считаем к3
  
    F(k4,y_k + k3*h, x + h ) #считаем к4
   
   
def count_Y(x,h):
    global k1, k2, k3, k4, y_new, y_k, y_k1
    runge(x,h)
    y_k1 = y_k + (h*(k1 + 2 * k2 + 2 * k3 + k4))/ 6 
    runge(x, h/2) #Началось мельчение
    y_mid = y_k + (h*(k1 + 2 * k2 + 2 * k3 + k4))/ 12
    
    runge(x+h/2, h/2)
    y_new = y_mid + (h*(k1 + 2 * k2 + 2 * k3 + k4))/12 #результат с половинчатым шагом
    norma(y_k1, y_new)
    
k = 0 
x = 0
y_k[0] = 1
y_new[0] = 0
#def autostep():
  #  global k1, k2, k3, k4, y_new, y_k, y_k1,x,h
while True:
    
    count_Y(x,h)
    print("yk1", y_k1)
    print("norma", n)
    
    
    while (n>EPS_MAX):
        print("большая норма")
        h = h*0.5
        if(h<h_min):
            
            sys.exit(0)
        count_Y(x,h)
    x = x + h
    if(x>=leng):
        x = x - h
        h = leng - x
        print("x = %.2Le h = %.7Le" %(x, h), file = my)
        print("Y_K =  %.8Le" %(y_k), file = my)
        count_Y(x,h)# Считаем настоящее значение в крайней правой точке xN
        x = leng
        print ("ПОСЛЕДНЯЯ ТОЧКА x = %Le y_k = %Le y(x) = %Le del = %Le h = %Le" %(x, y_k1, y_true(x), fabs(y_true(x)-y_k1), h), file = my)   
        print ("%Le %Le %Le %Le %Le" %(x, y_k1, y_true(x), fabs(y_true(x)-y_k1), h), file = my_gr)
        x = leng
        y_true(x)
            
        break
    y_true(x)
    y_k = y_k1 
    
    print(y_k1)
  #  print("должно быть", y_true(x))
   # print("получилось", y_k1)
   #s print("\n")
    print ("x = %Le y_k = %Le y(x) = %Le del = %Le h = %Le" %(x, y_k1, y_true(x), fabs(y_true(x)-y_k1), h), file = my)   
    print ("%Le %Le %Le %Le %Le" %(x, y_k1, y_true(x), fabs(y_true(x)-y_k1), h), file = my_gr)    
    if n < EPS_MIN:
        h = h * 2
        if h > h_max:
            h = h_max
    k = k + 1
print ("K=", k)

       	

