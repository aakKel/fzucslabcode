# 文章名：An Improved Approximation Algorithm for the
# Column Subset Selection Problem
import numpy as np

from ksvdPack import *
from pack import *
import heapq


def get(sensing,choice_list):

    j = 0
    for i in range(0,len(choice_list)):
        if choice_list[i] == 1:
            sensing[j][i] = 1
            j = j + 1
    return sensing

def get_k_col_matrix(sensing_matrix,k):
    u, s, v_t = np.linalg.svd(sensing_matrix,full_matrices=False)
    print(u.shape,s.shape,v_t.shape)
    s_r = np.zeros([s.shape[0],s.shape[0]])
    for i in range(0,s.shape[0]):
        s_r[i][i] = s[i]
    #print("shape:",u.shape,s_r.shape,v.shape)
    v = v_t.T
    v_k = v
    print(v_k.shape)
    v_k_l = sensing_matrix.shape[1]
    # 公式左下
    leftbottom = 0
    for j in range(0,v_k_l):
        leftbottom = leftbottom + np.linalg.norm(v_k[j:j+1,:])
    print(leftbottom)
    pro = np.zeros(v_k_l)
    for i in range(0,v_k_l) :
        pro[i] =(0.5 * np.linalg.norm(v_k[i:i+1,:]))/leftbottom


    max_indexs = heapq.nlargest(k, range(len(pro)), pro.take)
    max_indexs = np.sort(max_indexs)
    print("choice_col",max_indexs)
    sensing_matrix_res = np.zeros([sensing_matrix.shape[0],sensing_matrix.shape[1]])

    choice_list = np.zeros(sensing_matrix.shape[1])
    for i in range(0,len(max_indexs)):
        choice_list[max_indexs[i]] = 1
    # 转化为0/1
    sensing_matrix_res = get(sensing_matrix_res,choice_list)
    # 不转，直接为0列置0
    # for i in range(0,len(max_indexs)):
    #     sensing_matrix_res[:,max_indexs[i]] = sensing_matrix[:,max_indexs[i]]
    #print(sensing_matrix_res)
    return sensing_matrix_res






if __name__ == '__main__':
    sensing_matrix = GetGaussianMtx(5,20)
    get_k_col_matrix(sensing_matrix,5)
