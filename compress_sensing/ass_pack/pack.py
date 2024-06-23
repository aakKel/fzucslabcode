#import cupy as np
import numpy as np
#from PIL import Image
# from scipy.fftpack import fft,ifft
# import matplotlib.pyplot as plt
# from matplotlib.pylab import mpl
from sklearn import linear_model
import pandas as pd
import math 
from scipy.linalg import toeplitz
from scipy.fftpack import fft,ifft
import random


def my_omp(y,A,K,L):
    cols=A.shape[1]#���о���A������ 
    res=y #��ʼ���в�r0 ֵΪy
    indexs=[]#������������������
    A_c=A.copy()#���о���A�Ŀ���
    #L Ϊ��������
    #����K�ε���
    for i in range(0,K):
        products=[]#��������ÿ�ε����������ڻ�
        #���ڴ��о���A��ÿһ�н��м���
        for col in range(cols):
            #���о���A��col����в���ڻ�    
            products.append(np.dot(res,A[:,col]))#���һ���ڻ� ����products����

        index=np.argmax(np.abs(products)) # ÿ�м�����ɺ� ��products������ڻ�������������ֵ    
        
        indexs.append(index)#�����������ֵ������������indexs[]
        #ʹ���������ڴ��о����л���Ӽ�
        inv=np.dot(A_c[:,indexs].T,A_c[:,indexs])

        theta=np.dot(np.dot(np.linalg.inv(inv),A_c[:,indexs].T),y)#������С���˹��� ����һ�Φ�
        # print(theta.shape)
        res=y-np.dot(A_c[:,indexs],theta)#���²в�
    
    theta_final=np.zeros(L,)#�ؽ�theta ���ö�Ӧ������
    theta_final[indexs]=theta
    return theta_final


def Omp(y,A,K,L):
    cols=A.shape[1]#���о���A������ 
    res=y #��ʼ���в�r0 ֵΪy
    indexs=[]#������������������
    A_c=A.copy()#���о���A�Ŀ���
    #L Ϊ��������
    #����K�ε���
    theta = 0
    for i in range(0,K):
        products=[]#��������ÿ�ε����������ڻ�
        #���ڴ��о���A��ÿһ�н��м���
        for col in range(cols):
            #���о���A��col����в���ڻ�    
            products.append(np.dot(A[:,col].T,res))#���һ���ڻ� ����products����

        index=np.argmax(np.abs(products)) # ÿ�м�����ɺ� ��products������ڻ�������������ֵ    
        
        indexs.append(index)#�����������ֵ������������indexs[]
        #ʹ���������ڴ��о����л���Ӽ�
        inv=np.dot(A_c[:,indexs].T,A_c[:,indexs])#

        theta=np.dot(np.dot(np.linalg.inv(inv),A_c[:,indexs].T),y)#������С���˹��� ����һ�Φ�
        # print(theta.shape)
        res=y-np.dot(A_c[:,indexs],theta)#���²в�
    
    
    theta_final=np.zeros(L,)#�ؽ�theta ���ö�Ӧ������
    theta_final[indexs]=theta
    return theta_final

def GetSparseRandomMtx(M,N,d):
    Phi=np.zeros((M,N))
    for col in range(N):
        indexs=np.arange(M)
        np.random.shuffle(indexs)
        Phi[indexs[0:d],col]=1
    
    return Phi
    
def GetGaussianMtx(M,N):
    return np.random.randn(M,N)
def GetGaussianMtx_whk(M,N):
    # return np.random.randn(M,N)/math.sqrt(M)
    return np.random.normal(loc = 0,scale = 1/M,size = [M,N])
def InitD_norm(M,N):
    D = np.random.normal(loc = 0,scale = 1/M,size = [M,N])
    D_normed = D / D.max(axis=0)
    return D_normed
#���ԽǾ���[������-1,2,-1]

def trip1(m):
    Psi = np.eye(m)
    for i in range(m):
        if i < m - 1 and i > 0:
            Psi[i][i+1] = -1
            Psi[i][i-1] = -1
        elif i == 0:
            Psi[i][i+1] = -1
        elif i == m - 1:
            Psi[i][i-1] = -1
        Psi[i][i] = 2
    return np.linalg.inv(Psi)

def trip2(m):
    Psi = np.eye(m)
    for i in range(m):
        if i < m - 1:
            Psi[i][i+1] = -1
    return np.linalg.inv(Psi)

def my_dct(m):
    Psi = np.zeros([m,m])
    for k in range(1,m):
        for n in range(1,m):
            Psi[k,n] = np.cos((2*n-1)*(k-1)*math.pi/(2*m))
        if k==1:
            Psi[k,:] = math.sqrt(1/m) * Psi[k,:]
        else:
            Psi[k,:] = math.sqrt(2/m) * Psi[k,:]
    return Psi.T

def my_Bernoulli(m,n):
    Psi = 2*np.random.randint(0,2,size=([m,n]))-1
    Psi = Psi / math.sqrt(m)
    return Psi


def creat_Hadmard(m, n):
    temp = m & n
    result = 0
    for step in range(4):
        result += ((temp >> step) & 1)
    if 0 == result % 2:
        sign = 1
    else:
        sign = -1
    return sign


def my_Hadmard(m, n):
    Row_Matrix = m
    Column_Matrix = n
    Hadmard = np.ones((Row_Matrix, Column_Matrix), dtype = np.float32)
    for i in range(Row_Matrix):
        for j in range(Column_Matrix):
            Hadmard[i][j] = creat_Hadmard(i, j)
    return Hadmard

def my_toeplitz(m,n): #�����������ľ���
    tmp = np.random.rand(n)
    Psi = toeplitz(tmp)
    Psi = Psi[:m,:]
    return Psi
def my_fft(m,n):
    Psi = fft(np.eye(n))
    tmp = list(range(0,m))
    random.shuffle(tmp)
    Psi = Psi[tmp,:]
    return Psi

def my_measure_dct(m,n):
    Psi = my_dct(n).T
    tmp = list(range(0,m))
    random.shuffle(tmp)
    Psi = Psi[tmp,:]
    return Psi

def train_my_norm(data):
    # ��һ��
    max = 0
    min = 1000
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if(data[i][j] > max):
                max = data[i][j]
            if(data[i][j] < min):
                min = data[i][j]
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            data[i][j] = (data[i][j] - min)/(max-min)
    return data

def list_norm(data):
    max = 0
    min = 1000
    for i in range(len(data)):
        if (data[i] > max):
            max = data[i]
        if (data[i] < min):
            min = data[i]
    for i in range(len(data)):
        data[i] = (data[i] - min) / (max - min)
    return data


def my_norm(data):
    return data
    # return train_my_norm(data)


def calMAPE(res_data,sampleMatrix,data):
    sample_vector = np.zeros(sampleMatrix.shape[1])
    for i in range(0,sampleMatrix.shape[0]):
        for j in range(0,sampleMatrix.shape[1]):
            if sampleMatrix[i][j] == 1:
                sample_vector[j] = 1
    mape = 0
    miss_num = 0
    for i in range(0,len(sample_vector)):
            if sample_vector[i] == 0:
                if data[i] == 0:
                    continue
                miss_num = miss_num + 1
                mape = mape + np.abs(data[i] - res_data[i]) /data[i]
    mape = (mape / miss_num) * 100
    return mape
def calMAE(res_data,sampleMatrix,data):
    sample_vector = np.zeros(sampleMatrix.shape[1])
    for i in range(0,sampleMatrix.shape[0]):
        for j in range(0,sampleMatrix.shape[1]):
            if sampleMatrix[i][j] == 1:
                sample_vector[j] = 1
    mae = 0
    miss_num = 0
    for i in range(0,len(sample_vector)):
            if sample_vector[i] == 0:
                if data[i] == 0:
                    continue
                miss_num = miss_num + 1
                mae = mae + np.abs(data[i] - res_data[i])
    mae = mae / miss_num
    return mae
def calRMSE(res_data,sampleMatrix,data):
    sample_vector = np.zeros(sampleMatrix.shape[1])
    for i in range(0,sampleMatrix.shape[0]):
        for j in range(0,sampleMatrix.shape[1]):
            if sampleMatrix[i][j] == 1:
                sample_vector[j] = 1
    rmse = 0
    miss_num = 0
    for i in range(0,len(sample_vector)):
            if sample_vector[i] == 0:
                if data[i] == 0:
                    continue
                miss_num = miss_num + 1
                rmse = rmse + (data[i] - res_data[i])*(data[i] - res_data[i])
    rmse =  rmse / miss_num
    rmse = math.sqrt(rmse)
    return rmse

# ��ȡ�в�
def getErrorlist(resError,datasize,getEdata,Psi,finalMatrix,L):
    for fi in range(datasize):
        #getEdata Ϊԭʼ���� ����ΪL
        x_cal_error_tmp_ori = getEdata[fi]
        y_cal_error_tmp_sample = np.dot(finalMatrix,x_cal_error_tmp_ori)
        A = np.dot(finalMatrix,Psi)
        theta_cal_error_tmp_final = linear_model.orthogonal_mp(A, y_cal_error_tmp_sample)
        #x = psi . s
        x_cal_error_tmp_res = np.dot(Psi,theta_cal_error_tmp_final)

        for fj in range(L):
            resError[fj] = resError[fj]  + x_cal_error_tmp_ori[fj]- x_cal_error_tmp_res[fj]
    for fj in range(L):
        resError[fj] = resError[fj] / datasize
    #print("resError:" ,resError)
    return resError


def recovery_cal_error(self, ori_x, ori_fi, m):
    '''
    论文出处：Joint Dictionary Learning and Recovery Algorithms in a Jointly Sparse Framework
    ori_x : 原始数据，待采集的，维度应当是 L*n,L为站点个数，n为单条序列长度
    ori_fi : 应当是 L 个 m * n的矩阵拼起来的，形如 行数为m，列数为L*n
    '''
    # 采样矩阵Y
    Y = []
    for i in range(0, self.num):
        Y.extend(np.dot(ori_fi[i], ori_x[i].T))
    A = ori_fi.copy()
    for i in range(0, self.num):
        A[i] = np.dot(A[i], self.Psi)
    # 把num个A拼成斜对角矩阵
    fi = np.block([[A[i] if i == j else np.zeros((m, self.L)) for j in range(len(A))] for i in range(len(A))])
    xk = np.dot(np.linalg.pinv(fi), Y)
    # ----------------------------------#
    tiny_error = 0.1  # 惩罚项      参数
    p = 0.5  # 0<= p <=1           参数
    min_error = 1  # 终止的误差限   参数
    LD = 0.1  # 考虑噪声误差         参数
    # ----------------------------------#
    ## 算法流程见论文
    while 1:
        diag_matrix = np.zeros([self.L, self.L])
        for i in range(0, self.L):
            diag_matrix[i][i] = np.power(np.sqrt(np.linalg.norm(np.sum(xk[i])) + tiny_error), 1 - p / 2)
        Wk = np.kron(np.eye(self.num), diag_matrix)
        Ak = np.dot(fi, Wk)
        Ak_tmp = np.dot(Ak, np.conj(Ak).T)
        new_I = np.multiply(LD, np.eye(Ak_tmp.shape[0]))
        new_xk = np.dot(np.dot(Wk, np.conj(Ak).T), np.dot((np.linalg.inv(np.add(Ak_tmp, new_I))), Y))

        if (np.linalg.norm(np.subtract(new_xk, xk)) / np.linalg.norm(xk)) < min_error:
            xk = new_xk
            break
        xk = new_xk

    # 得到的xk是维度是一维的，转化为2维
    res_x = np.reshape(xk, (self.num, self.L))
    res_tmp = pd.DataFrame(res_x)
    # res_tmp.to_csv('sparse.csv',sep=' ',index=0,header=1,mode='a')
    # print(res_x)
    # 此时的res_x 是稀疏的，乘上稀疏基
    for i in range(0, self.num):
        res_x[i] = np.dot(self.Psi, res_x[i])
    # 替换采样值
    cntt = 0
    for i in range(0, self.num):
        for j in range(0, m):
            for k in range(0, self.L):
                if ori_fi[i][j][k] == 1:
                    res_x[i][k] = ori_x[i][k]
                    cntt += 1
                    # 一行只有一个1 ,退出当前循环
                    break
    # print("采样个数",cntt)
    # 计算误差
    # mape,rmse
    mape = 0
    cnt = 0
    rmse = 0
    mae = 0
    for i in range(0, self.num):
        for j in range(0, self.L):
            if ori_x[i][j] == 0:
                mape += 1
                rmse += 1
                mae += 1
                continue
            cnt += 1
            mape += np.abs(ori_x[i][j] - res_x[i][j]) / ori_x[i][j]
            rmse += (ori_x[i][j] - res_x[i][j]) ** 2
            mae += np.abs(ori_x[i][j] - res_x[i][j])

    mape = (mape / cnt) * 100
    mape = round(mape, 3)
    rmse = rmse / cnt
    rmse = math.sqrt(rmse)
    rmse = round(rmse, 3)
    mae = mae / cnt
    mae = round(mae, 3)

    if mape > 100:
        mape = 100
    print("当前采样个数：", cntt, "实际计算的个数", cnt, "缺失率：", round(100 - cntt / cnt * 100, 3), "mape为：", mape, "  rmse 为:", rmse,
          "  mae为:", mae)

    ## 画图
    # print(res_x)
    # print(ori_x)
    # x_line = np.linspace(0, ori_x.shape[1] - 1, ori_x.shape[1])
    # for pltidx in range(0,ori_x.shape[0]) :
    #     fig = plt.figure(figsize=(12, 14), dpi=300)
    #     plt.title("Sample rate : " + str(m) + "    No:" + str(pltidx + 1))
    #     plt.plot(x_line, ori_x[pltidx], c='black', ls='-', alpha=1, label='ori_x')
    #     plt.plot(x_line, res_x[pltidx], c='red', ls='-', alpha=1, label='res_x')
    #     plt.show()
    return mape, rmse, mae, Y