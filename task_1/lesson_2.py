
import matplotlib.pyplot as ppl
import math
import numpy as np
def f(x):
    out = 4*r*x*(1-x)
    return out

def f_d(x):
    out = 4*r - 8*r*x
    return out



r = 0
x_0 = 0.5
delta = 0.01
x_d = x_0 + delta
res = []
coor = []
n = 250
z = []
while r<1:
    x = x_0
    x_d = x_0 + delta
    for i in range (n):
        x = f(x)
        x_d = f(x_d)
        if i == n-1:
            res.append(1/n * np.log(abs(x_d - x)/delta))
            coor.append(r)
            z.append(0)
    r += 0.001

s = [0.1 for n in range(len(res))]
fig, ax = ppl.subplots(1, 1)
ppl.scatter(coor, z, s=s)
ppl.scatter(coor,res,s=s)
ppl.show()



# r = 0
# x_0 = 0.5
# delta = 0.0001
# x_d = x_0 + delta
# res = []
# coor = []
# n = 300
# res_f = 1
# z = []
# while r<1:
#     x = x_0
#     x_d = x_0 + delta
#     res_f = 1
#     for i in range (n):
#         x = f(x)
#         res_f *= abs(f_d(x))
#     res.append(np.log(res_f)/n)
#     coor.append(r)
#     z.append(0)
#     r += 0.001

# s = [0.1 for n in range(len(res))]
# fig, ax = ppl.subplots(1, 1)
# ppl.scatter(coor, z, s=s)
# ppl.scatter(coor,res,s=s)
# ppl.show()
