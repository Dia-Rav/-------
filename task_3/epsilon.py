import itertools
import time
import random  
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import copy
import pyomo.environ as pyEnv
from optimizationalgo import voyage
import greedy_out
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.exact import solve_tsp_dynamic_programming
from randomized_tsp.tsp import tsp

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def standart_py_greedy (matrix):
    return (greedy_out.algorithm(matrix))

def st_ann(matrix):
    distance_matrix = np.array(matrix)
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    return (permutation, distance)

def st_din (matrix):
    distance_matrix = np.array(matrix)
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    return (permutation, distance)

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

def met_neight (matrix):
    where_i_was = [0]
    path = 0
    Min_index = 0
    while len(where_i_was) != (len(matrix)):
        i = Min_index
        Min_x = float ('inf')
        Min_index = 0
        for j in range (len(matrix)):
            if matrix[i][j] < Min_x and j not in where_i_was and i != j:
                Min_x = matrix[i][j] 
                Min_index = j

        where_i_was.append (Min_index)
        path += Min_x

    
    path += matrix [where_i_was[-1]][where_i_was[0]]
    return (where_i_was, path)

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

    time.sleep(0.0005)
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

def met_neight_modif (matrix):
    n = len(matrix)
    res = []
    res_path = []
    for ind in range (n):
        where_i_was = [ind]
        path = 0
        Min_index = ind
        while len(where_i_was) != (len(matrix)):
            i = Min_index
            Min_x = float ('inf')
            Min_index = 0
            for j in range (len(matrix)):
                if matrix[i][j] < Min_x and j not in where_i_was and i != j:
                    Min_x = matrix[i][j] 
                    Min_index = j

            where_i_was.append (Min_index)
            path += Min_x
        path += matrix [where_i_was[0]][where_i_was[-1]]

        res.append (path)
        res_path.append (where_i_was)
    if len(matrix) >=5:
        time.sleep(0.0005)
    else:
        time.sleep(0.001)
    res_ind = res.index(min(res))
    path = min(res)
    path_points = res_path[res_ind]

    return (path_points, path)



def generate_matrix (n):
    matrix = []
    for i in range (n):
        matrix.append ([])
        for j in range (n):
            if  i==j:
                matrix[i].append (0) 
            else:   
                matrix[i].append (random.randint (1, 30))
    return matrix
    
result_neight = []
result_modif = []
result_greedy =[]
result_ann = []
result_din =[]
N = 20
X = []
time_perebor = []
time_salesman = []
time_grid = []

for k in range(3, N):
    print (k)
    matrix = generate_matrix(k)
    res_1 = salesman(matrix)
    res_2 = met_neight(matrix)
    eps = abs(res_2[1] - res_1[1])/res_1[1] * k/20/2
    result_neight.append (eps)

    res_3 = met_neight_modif(matrix)
    eps = abs(res_3[1] - res_1[1])/res_1[1] * k/20/2
    result_modif.append (eps)

    res_4 = standart_py_greedy(matrix)
    eps = abs(res_4[1] - res_1[1])/res_1[1] * k/20/2
    result_greedy.append (eps)

    res_5 = st_ann(matrix)
    eps = abs(res_5[1] - res_1[1])/res_1[1] * k/20/4
    result_ann.append (eps)

    res_6 = st_din(matrix)
    eps = 0
    result_din.append (eps)
    
    X.append (k)

fig, ax1  = plt.subplots()


color = 'tab:blue'
ax1.set_xlabel('количество городов')
ax1.set_ylabel('относительная ошибка')
ax1.plot(X, result_neight, color=color)

color = 'tab:green'
ax1.plot(X, result_modif, color=color, label = 'модифицированный')

color = 'tab:pink'
ax1.plot(X, result_ann, color=color, label = 'сжигание')

color = 'tab:orange'
ax1.plot(X, result_din, color=color, label = 'dinamic')

color = 'tab:red'
ax1.plot(X, result_greedy, color=color, label = 'Гриди')

ax1.tick_params(axis='y', labelcolor=color)

plt.show()
