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
p = 0.2 #вероятность заражения

days  = 30
coor = []
res = []
n_1 = 1
n_2 = 0
res_1= []
res_2 = []
n = 1
for day in range (days):
    temp = n_1
    n_1 = f(n_1, n_2, p)
    n_2 = temp
    res.append(n_1 + n_2)
    coor.append(day)
    res_1.append(n_1)
    res_2.append(n_2)



s = [1 for i in range (len(coor))]
# plt.plot(coor,res)
plt.plot(coor,res_1, 'g')
plt.plot(coor,res_2, 'y')

plt.show()
