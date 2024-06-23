# 文章名：An Improved Approximation Algorithm for the
# Column Subset Selection Problem
import numpy as np
import heapq


def get(sensing,choice_list):

    j = 0
    for i in range(0,len(choice_list)):
        if choice_list[i] == 1:
            sensing[j][i] = 1
            j = j + 1
    return sensing

def get_k_col_matrix(sensing_matrix,k):
    u, s, v = np.linalg.svd(sensing_matrix,full_matrices=False)
    # print(u.shape,s.shape,v.shape)
    s_r = np.zeros([s.shape[0],s.shape[0]])
    for i in range(0,s.shape[0]):
        s_r[i][i] = s[i]
    #print("shape:",u.shape,s_r.shape,v.shape)

    # 取前K行
    v_k = v[:k,:]
    v_k_l = sensing_matrix.shape[1]
    # print(v_k)
    # 公式左下
    leftbottom = 0
    for j in range(0,v_k_l):
        leftbottom = leftbottom + np.linalg.norm(v_k[:,j:j+1])

    p_k = s.shape[0] - k
    # print(p_k)
    # 公式右下
    rightbottom = 0
    e_v_k = np.dot(s_r[p_k:,p_k:],v[p_k:,:])
    # print(e_v_k)
    for j in range(0,v_k_l):
        rightbottom = rightbottom + np.linalg.norm(e_v_k[:,j:j+1])

    pro = np.zeros(v_k_l,dtype=np.float64)
    for i in range(0,v_k_l):
        pro[i] = (0.5 * np.linalg.norm(v_k[:,i:i+1]))/leftbottom + (0.5 * np.linalg.norm(e_v_k[:,i:i+1]))/rightbottom

    max_indexs = heapq.nlargest(k, range(len(pro)), pro.take)
    max_indexs = np.sort(max_indexs)
    # print("当前单站点采样个数:",len(max_indexs))
    # print("choice_col",max_indexs)
    sensing_matrix_res = np.zeros([sensing_matrix.shape[0],sensing_matrix.shape[1]])

    choice_list = np.zeros(sensing_matrix.shape[1])
    for i in range(0,len(max_indexs)):
        choice_list[max_indexs[i]] = 1

    #-------------------#
    # for i in range(len(choice_list)):
    #     if choice_list[i] == 0:
    #         print('-', end="")
    #     else:
    #         print('|', end="")
    # print()
    # 转化为0/1
    sensing_matrix_res = get(sensing_matrix_res,choice_list)
    # 不转，直接为0列置0
    # for i in range(0,len(max_indexs)):
    #     sensing_matrix_res[:,max_indexs[i]] = sensing_matrix[:,max_indexs[i]]
    #print(sensing_matrix_res)
    return sensing_matrix_res


