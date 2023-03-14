import matplotlib.pyplot as plt
import math
import numpy as np
import random 
from statistics import fmean as fmean
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf

def f(n, p):
    out = p*n
    return out

def avg_arr (arr):
    sum = 0
    for i in range (len(arr)):
        sum+=arr[i]
    return (sum/len(arr))

N = 1000 #население
n = 1 #больных
p = 0.00001 #вероятность заражения
K = 10

days  = 500
coor = []
res = []
res_1 = []
while p<=0.0013:
    res_1 = []
    n = 1
    for day in range (days):
        n_1 = 0
        p_x = f(n, p)
        sum = 0
        for k in range(K):
            n_1 = 0
            for man in range (N-n):
                x = random.random()
                if x <= p_x:
                    n_1 +=1
            sum+=n_1
        n_1 = sum/K
        n = round(n_1)
        if day >= days-30:
            res_1.append(n_1)
    
    coor.append(p)
    print (res_1, p)
    res.append(avg_arr(res_1))
    p+=0.00001
s = [1 for i in range (len(coor))]
plt.scatter(coor,res, s = s)
plt.show()


#0.00105   1- (1-p)^n
#0.00109   pn