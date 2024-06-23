import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from two_page_paper import get_k_col_matrix
import math
from sklearn import linear_model
import pandas as pd

import time


import random
import math


def distance(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # 地球半径，单位为千米
    r = 6371

    # 计算经纬度距离
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = r * c

    return distance
def Oth(lat1, lon1, lat2, lon2) :
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def bit_cnt(num,mx):
    res = 0
    a_list = []
    for i in range(mx):
        if (num >> i) & 1 == 1:
            res += 1
            a_list.append(i)
    return res,a_list

def cal(data,s_num,a,b) :
    mx = 1 << 24
    dist = np.zeros([s_num,s_num])
    for i in range(s_num):
        for j in range(s_num):
            # 自身不可达
            if i == j :
                dist[i][j] = 1e6
            else:
                dist[i][j] = distance(data[i][0],data[i][1],data[j][0],data[j][1])
    res_a = []
    res_b = np.zeros([a,b],dtype=np.int)
    sum = 5e6
    for i in range(mx):
        bit_num,a_list = bit_cnt(i,s_num)
        if bit_num != a :
            continue
        # b_list = list(range(0, s_num))
        dist_tmp = dist.copy()
        # 把邻居b去除a
        for j in range(a):
            for k in range(s_num):
                if k in a_list:
                    dist_tmp[a_list[j]][k] = 1e6
        print(i,a_list)
        choice_b = np.zeros([a,b],dtype=np.int)
        sum_t = 0
        for j in range(a):
            a_tmp = np.argsort(dist_tmp[a_list[j]])
            choice_b[j] = a_tmp[:b]
            for k in range(b):
                sum_t += dist_tmp[a_list[j]][choice_b[j][k]]
        print(choice_b)
        print(sum_t)
        if sum_t < sum :
            sum = sum_t
            res_a = a_list
            res_b = choice_b
    return res_a,res_b

# 示例
p = '../dataset/36/pm25_latlng.csv'
data = pd.read_csv(p)
data = data.iloc[:, 1:]
data = data.values.astype(np.float64)
a = 13
b = 4
s_num = 36
start_time = time.time()
res_a,res_b = cal(data,s_num,a,b)
end_time = time.time()
run_time = end_time - start_time
print(res_a)
print(res_b)
print(run_time)

