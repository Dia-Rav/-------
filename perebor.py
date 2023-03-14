import itertools
import time
import random  
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import copy

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def Min(lst,myindex):
    return min(x for idx, x in enumerate(lst) if idx != myindex)


#функция удаления нужной строки и столбцах
def Delete(matr,index1,index2):
    matrix_1 = copy.deepcopy(matr)
    del matrix_1[index1]
    for row in matrix_1:
        del row[index2]

    return matrix_1

def minus_min(matrix):
        H_tmp = 0
    
        for i in range(len(matrix)):
            min_row = min(matrix[i])
            H_tmp += min_row 
            for j in range(len(matrix)):
                matrix[i][j] -= min_row
        for i in range (len(matrix)):    
            min_column = min(row[i] for row in matrix)
            H_tmp += min_column
            for j in range (len(matrix)):
                matrix[j][i] -= min_column
        return (matrix, H_tmp)

#Функция вывода матрицы
def PrintMatrix(matrix):
    print("---------------")
    for i in range(len(matrix)):
        print(matrix[i])
    print("---------------")


def salesman (matrix):
    n=len(matrix)
    H=0
    PathLenght=0
    Str=[]
    Stb=[]
    result = []
    result=[]
    StartMatrix=[]

    #Инициализируем массивы для сохранения индексов
    for i in range(n):
        Str.append(i)
        Stb.append(i)
        
    #Сохраняем изначальную матрицу
    StartMatrix = copy.deepcopy(matrix)

    #Присваеваем главной диагонали float(inf)
    for i in range(n): 
        matrix[i][i]=float('inf')

    while True:
        res = minus_min(matrix)
        matrix = res[0]
        H = res[1]
        
        #Оцениваем нулевые клетки и ищем нулевую клетку с максимальной оценкой
        #--------------------------------------
        NullMax=0
        index1=0
        index2=0
        tmp=0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i][j]==0:
                    tmp=Min(matrix[i],j)+Min((row[j] for row in matrix),i)
                    if tmp>=NullMax:
                        NullMax=tmp
                        index1=i
                        index2=j

        #если включаем найденную точку
        matrix_tmp = copy.deepcopy(matrix)
        if Str[index1] in Stb and Stb[index2] in Str:
            ind1 = Stb.index (Str[index1])
            ind2 = Str.index (Stb[index2])
            matrix_tmp[ind2][ind1] = float ('inf')
        matrix_tmp = Delete(matrix_tmp, index1,index2)
        res = minus_min(matrix_tmp)
        matrix_tmp_1 = res[0]
        H_tmp_1 = res[1]
        
        #если не включаем
        matrix_tmp = copy.deepcopy(matrix)
        matrix_tmp[index1][index2] = float('inf')
        res = minus_min(matrix_tmp)
        matrix_tmp_2 = res[0]
        H_tmp_2 = res[1]

        

        if H_tmp_1 < H_tmp_2:
            matrix = matrix_tmp_1
            H+=H_tmp_1
            result.append((Str[index1]+1, Stb[index2]+1))
            del Str[index1]
            del Stb[index2]
        else:
            matrix = matrix_tmp_2
            H += H_tmp_2
        print (matrix, Str, Stb)
        if len(matrix)== 2:

            if matrix[0][0] != float('inf') and matrix[1][1] != float('inf'):
                result.append((Str[0]+1, Stb[0] + 1))
                result.append((Str[1]+1, Stb[1] + 1))
            else:
                result.append((Str[0]+1, Stb[1] + 1))
                result.append((Str[1]+1, Stb[0] + 1))
            break


    path = 0
    for step in result:
        path += StartMatrix[step[0]-1][step[1]-1]
    res = [result[0]]
    while len(res) != len(result):
        for step in result:
            if res[-1][1] == step [0]:
                res.append (step)
                break


    return (res, path)


def perebor (matrix):
    path = float('inf')
    path_x = ()
    tmp = ''
    for i in range (len(matrix)):
        tmp += str(i)
    for way in itertools.permutations(tmp, len(matrix)):
        temp = 0
        for i in range(len(way)-1):
            temp += matrix[int(way[i])][int(way[i+1])]
        temp += matrix[int(way[-1])][int(way[0])]
        if temp<= path:
            path = temp
            path_x = way
    path_x_correct = []
    for i in range (len(path_x)):
        path_x_correct.append(int(path_x[i])+1)
    return (path_x_correct, path)

time_perebor = []
time_salesman = []

def generate_matrix (n):
    matrix = []
    for i in range (n):
        matrix.append ([])
        for j in range (n):
            if  i==j:
                matrix[i].append (0) 
            else:   
                matrix[i].append (random.randint (0, 20))
    return matrix
    

N = 60
X = []

# matrix =[[0, 7, 2, 9, 7], [5, 0, 3, 9, 1], [4, 8, 0, 5, 3], [5, 6, 4, 0, 7], [7, 6, 3, 7, 0]]
# print(perebor (matrix))
# print (salesman(matrix))

for k in range(3, N):
    print (k)
    matrix = generate_matrix(k)
    # start_time = time.time()
    # print(perebor (matrix))
    # end_time = time.time() - start_time
    # time_perebor.append(end_time)

    start_time = time.time()
    print (salesman(matrix))
    end_time = (time.time() - start_time)
    time_salesman.append(end_time)
    
    X.append (k)


fig, ax1  = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('количество городов')
# ax1.set_ylabel('время перебора', color=color)
# ax1.plot(X, time_perebor, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax1.set_ylabel('время граней и границ', color=color)
ax1.plot(X, time_salesman, color=color)
ax1.tick_params(axis='y', labelcolor=color)

plt.show()
