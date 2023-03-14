import matplotlib.pyplot as plt
import math
import numpy as np
import random 
from statistics import fmean as fmean
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf


def f(x):
    out = x**2
    return out


n_1 = 0
N = 1100 #сколько точек выбираем
n = 50 #сколко раз бросаем одинаковое кол-во камней
s = 8/3 #реальная площадь
S = 8 #площадь прямоугольника
K = 200 #сколько раз меняем N
res = []
coor = []
res_1 = []
for k in range (K):
    for j in range (n):
        d = []
        n_1 = 0 #сколько раз мы попали под график
        for i in range (N):
            x = random.uniform(0, 2)
            y = random.uniform(0, 4)
            if y<f(x):
                n_1+=1

        s_1 = S*n_1/N #оцененная площаль (или интеграл)
        
        delta = abs(s - s_1)/s #относительная погрешность
        d.append(delta)
    d_avg = fmean(d) #среднее значение относительной погрешности для каждого N
    res.append(np.log(d_avg))
    coor.append(np.log(N))
    res_1.append(-1/2*np.log(N))

    
    N+=50
res = np.array(res)
coor = np.array(coor)


fig, ax = plt.subplots(1, 1)
df = pd.DataFrame({'x':coor,'y':res})
sns.lmplot(x='x',y='y', data=df, order=1)
plt.scatter(coor,res)
plt.plot(coor, res_1)
plt.show()



model = smf.ols("y ~ x", data=df)
model_est = model.fit()
print(model_est.summary())
#k = -0.4892     p = 0.001 
#k_0 = -1.0699   p = 0.319
