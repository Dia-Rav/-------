import matplotlib.pyplot as plt
import math
import numpy as np
import random 
from statistics import fmean as fmean
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf

def f(n_1, n_2, p):
    n = n_1 + n_2
    return (1-(1-p)**n)*(N-n)

def avg_arr (arr):
    sum = 0
    for i in range (len(arr)):
        sum+=arr[i]
    return (sum/len(arr))

N = 1000 #население
p = 0.0001 #вероятность заражения

days  = 500
coor = []
res = []
while p<=0.002:
    res_1 = []
    n = 1
    n_1 = 1
    n_2 = 0
    for day in range (days):
        temp = n_1
        n_1 = f(n_1, n_2, p)
        n_2 = temp
        if day >= days-30:
            res_1.append(n_1 + n_2)
    
    coor.append(p)
    res.append(avg_arr(res_1))
    p+=0.00001

N = 1000 #население
p = 0.0001 #вероятность заражения
days  = 500
res_1_day = []
res_1_day_1 = []
while p<=0.002:
    res_1_day_1 = []
    n = 1
    for day in range (days):
        n = f(n, 0, p)
        if day >= days-30:
            res_1_day_1.append(n)
    res_1_day.append(avg_arr(res_1_day_1))
    print (res_1_day)
    p+=0.00001

s = [1 for i in range (len(coor))]
plt.scatter(coor,res, s = s)
plt.scatter(coor,res_1_day, s = s)
plt.show()
