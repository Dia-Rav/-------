import matplotlib.pyplot as ppl
import math
# def f(x):
#     out = 4*r*x*(1-x)
#     return out


# res = []
# coor = []

# for i in range (1000):
#     x = 0.2
#     r = 0.001*(i+1)
#     for j in range( 200):
#         x = f(x)
#         if j >160:
#             res.append(x)
#             coor.append(r)

# s = [0.1 for n in range(len(res))]
# ppl.scatter(coor,res,s=s)
# ppl.show()

# r = [0.75 , 0.862372436 ,0.8860225 , 0.89110, 0.892175]
# # r = (1 + sqrt(6))/4

# for i in range (len(r)-2):
#     c = (r[i+1] - r[i])/(r[i+2] - r[i+1])
#     print (c)
# import matplotlib.pyplot as ppl

# def f(x):
#     out = x*math.e**(r*(1-x))
#     return out


# res = []
# coor = []
# r = 0.1

# while r<=4:
#     x = 0.2
#     r += 0.001
#     for j in range( 200):
#         x = f(x)
#         if j >160:
#             res.append(x)
#             coor.append(r)

# s = [0.01 for n in range(len(res))]
# ppl.scatter(coor,res,s=s)
# ppl.show()


# r = [2, 2.5252 , 2.6369, 2.66082]
# # 4.701880035810209
# # 4.6716854872437645

# for i in range (len(r)-2):
#     c = (r[i+1] - r[i])/(r[i+2] - r[i+1])
#     print (c)

def f(x):
    out = r*(1-(2*x-1)**4)
    return out


res = []
coor = []

r = 0.968
e = 10**(-6)
while r<0.9684:
    x = 0.2
    r += 0.00001
    X = []
    for j in range( 200):
        x = f(x)
        X.append (x)
        if j >160:
            res.append(x)
            coor.append(r)

s = [0.01 for n in range(len(res))]
ppl.scatter(coor,res,s=s)
# ppl.show()
r = [0.82701, 0.95093, 0.9663, 0.96821]


for i in range (len(r)-2):
    c = (r[i+1] - r[i])/(r[i+2] - r[i+1])
    print (c)

