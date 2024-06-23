# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import math
import random
from pack import *

def get_train_data(datasize,station,rate,day,p):
    '''
    datasize : 训练的原子个数，
    station : 站点编号
    rate : 缺失率
    day : 一次补几天
    '''

    station = station - 1
    datalen = 24
    L = datalen * datasize
    row = math.floor(datalen * day * rate)
    #p = 'AQI_beijing_mean.csv'
    data = pd.read_csv(p)
    data = data.iloc[(station*365):(station*365) + datasize* day,:]
    data = data.values.astype(np.float64)
    data = my_norm(data)
    y = np.zeros([datasize,day * datalen])
    final_data = np.zeros([datasize,row])
    index = 0
    for i in range(datasize):
        
        #np.random.seed(i+i)
        random_list = np.random.rand(datalen * day)
        # y 
        y_t = []
        for k in range(day):
            y_t.extend(data[index])
            index = index + 1
        y[i] = y_t
        tmpIndex = 0
        tmp = np.zeros(row)
        for j in range(len(random_list)):
            if tmpIndex >= row:
                break
            if random_list[j] <= rate:
                tmp[tmpIndex] = y[i][j]
                tmpIndex = tmpIndex + 1
        final_data[i] = tmp
    final_data = final_data.T
    return final_data,data

def get_train_data_Gussian(datasize,station,rate,day,beginDay,p):
    '''
    datasize : 训练的原子个数，
    station : 站点编号
    rate : 缺失率
    day : 一次补几天
    '''
    #station = station - 1
    datalen = 24
    L = datalen * day
    row = math.floor(datalen * day * rate)
    data = pd.read_csv(p)
    data = data.iloc[(station*365) + beginDay:(station*365) + beginDay + datasize* day,:]
    print("guss data from",(station*365) + beginDay,"to",(station*365) + beginDay + datasize* day)

    # print(data)
    data = data.values.astype(np.float64)
    data = my_norm(data).copy()
    y = np.zeros([datasize,day * datalen])
    final_data = np.zeros([datasize,row])
    oritraindata = np.zeros([datasize,day * datalen])
    idx = 0
    for i in range(datasize):
        
        #np.random.seed(i)
        gu = GetGaussianMtx_whk(row,L)
        gu = np.abs(gu)
        gu = to_01_matrix(gu,day,datalen)


        index = 0
        y_t = []
        for k in range(day):
            y_t.extend(data[index])
            index = index + 1
        oritraindata[idx] = y_t
        #y_t 为完整的day * 24 序列
        #oritraindata 为完整的原始训练数据
        # final_data为用高斯随机截取的字典学习训练数据
        idx = idx + 1
        y_r = np.dot(gu,y_t)
        final_data[i] = y_r
    final_data = final_data.T
    print(final_data.shape)
    #oritraindata = my_norm(oritraindata)
    return final_data,oritraindata


def to_01_matrix(sensingMatrix,day,datalen):
    row = sensingMatrix.shape[0]
    L = sensingMatrix.shape[1]
    finalMatrix = np.zeros([row,L],dtype=np.int32)
    cnt = 0
    i = 0
    round_time = 1
    while i < sensingMatrix.shape[0]:
        
        index = 0
        mx = 0
        for j in range(0,sensingMatrix.shape[1]):
            if sensingMatrix[i][j] > mx:
                mx = sensingMatrix[i][j]
                index = j
        f = False
        g = False
        # 查看选择的列是否是已经被选过
        for otherIndex in range(0,finalMatrix.shape[0]):
            if finalMatrix[otherIndex][index] == 1:
                f = True
                sensingMatrix[i][index] = 0
                break
        for otherIndex in range(0,finalMatrix.shape[1]):
            if sensingMatrix[i][otherIndex] > 0:
                g = True
                break
        #print(rate[r],sensingMatrix.shape,i,f,g)
        # 若当行为  1 的个数都被采集过，那么选择最少被采集的站点进行置1
        if g == False:
            sampled_tmp = np.zeros(day)
            for otherIndex in range(0,finalMatrix.shape[0]):
                for otherIndexJ in range(0,finalMatrix.shape[1]):
                    if finalMatrix[otherIndex][otherIndexJ] == 1:
                        sampled_tmp[math.floor(otherIndexJ / datalen)] += 1
            minstation = 100
            #print(len(sampled_tmp))
            for otherIndex in range(0,len(sampled_tmp)):
                if minstation > sampled_tmp[otherIndex]:
                    minstation = otherIndex
            #minstation = minstation * datalen
            station_zero = []
            for otherIndex in range(minstation,minstation+datalen):
                if finalMatrix[i][otherIndex] == 0:
                    station_zero.append(otherIndex)
            choice_rand = random.uniform(0,len(station_zero))
            choice_rand = math.floor(choice_rand)
            finalMatrix[i][station_zero[choice_rand]] = 1
            i = i + 1
            continue
        if f == True:
            continue
        finalMatrix[i][index] = 1
        cnt = cnt + 1
        i = i + 1
    return finalMatrix

def dict_update(y, d, x, n_components):

    for i in range(n_components):
        index = np.nonzero(x[i, :])[0]
        if len(index) == 0:
            continue
        d[:, i] = 0
        r = (y - np.dot(d, x))[:, index]
        u, s, v = np.linalg.svd(r, full_matrices=False)
        d[:, i] = u[:, 0]
        for j,k in enumerate(index):
            x[i, k] = s[0] * v[0, j]
    return d, x

def transtosort(sensingMatrix):
    row = sensingMatrix.shape[0]
    col = sensingMatrix.shape[1]
    sampleV = np.zeros(col)
    for i in range(row):
        for j in range(col):
            if sensingMatrix[i][j] == 1:
                sampleV[j] = 1
    res = np.zeros([row,col])
    h = 0
    for j in range(len(sampleV)):
        if sampleV[j] == 1:
            res[h][j] = 1
            h = h + 1
    return res