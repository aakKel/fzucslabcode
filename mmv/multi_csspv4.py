import random
from fromNumtoNpz import get_random_npz, get_greedy_npz
import numpy as np
from two_page_paper import get_k_col_matrix
import math
from sklearn import linear_model
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from pearsonr import get_pearsonr
import os
import  sys


# 2024.1.3
import datetime
#
is_load = True
is_load_rand = False
is_load_greedy = True
load_file_name = './output/npz/iter1000,n6output_20240104082110.npz'

class multi_cssp4:
    '''
    begin_day : 采样开始的时间点
    day ： 向量长度
    datasize : 训练集大小
    time : 采样轮数
    num : 站点个数
    '''

    def __init__(self, num, begin_day, day, datasize, time, rate, p, zero_p):
        self.num = num
        self.begin_day = begin_day
        self.day = day
        self.datasize = datasize
        self.time = time
        self.rate = rate
        self.p = p
        self.zero_p = zero_p
        self.datalen = 24
        self.L = self.day * self.datalen
        self.Psi = trip2(self.L)
        self.no_num = 5
        self.is_add = True
        self.near_s = 3


    def set_bit(self,x, k):
        return x | (1 << (k - 1))


    def load_from_file(self):
        filename = load_file_name
        if is_load_rand and is_load_greedy :
            print("is_load_rand and is_load_greedy error")
            print("exit")
            sys.exit()
        if is_load_rand:
            filename = get_random_npz(self.p, self.day, self.begin_day, self.num)
        if is_load_greedy:
            filename = get_greedy_npz(self.p, self.day, self.begin_day, self.num)
        data = np.load(filename)

        # 从data中读取矩阵A和B
        A = data['A'].tolist()
        B = data['B'].tolist()
        # os.remove(filename)
        return A,B

    def get_no_choice_station(self):
        # 相关性系数
        # pear_num : 站点相关性矩阵
        # conn_idx : 按行排序， 内容为下标
        pear_num, conn_idx = get_pearsonr(self.p, self.day, self.begin_day, self.num, False)
        pear_num, conn_idx = get_pearsonr(self.p, self.day, self.begin_day, self.num, True)
        # 遗传算法
        ## 参数
        # 字典学习迭代次数
        max_iter = 20
        tolerance = 1e-7
        n_nonzero = 20
        ## 遗传算法迭代次数
        max_iter_yc = 1000
        max_iter_yc_tmp = max_iter_yc
        # 变异概率
        variation_rate = 0.5
        # 交叉概率
        cross_rate = 0.5
        # 染色体数量
        N = 6
        # 训练以30 采样率进行训练
        rate = 0.3
        # 领居数量
        nb_num = self.near_s
        # 2进制表示 基因序列，最顶层为 up
        up = (1 << self.num) - 1
        min_up = (1 << (int)(self.num / 2)) - 1

        # 初始解数量
        solve_list = [0] * N
        for i in range(N):
            solve_list[i] = random.randint(0, up)
        # 初始种群交叉变异
        new_solve_list = [0] * N
        # 交叉变异各一半
        cross_num = math.floor(N / 2)
        for i in range(cross_num):
            a = random.randint(0, N - 1)
            b = random.randint(0, N - 1)
            new_solve_list[i] = self.cross(solve_list[a], solve_list[b], cross_rate)
        for i in range(cross_num, N):
            a = random.randint(0, N - 1)
            new_solve_list[i] = self.variation(solve_list[a], variation_rate)
        all_solve_list = solve_list + new_solve_list

        mn_error = 100
        mn_error_num = 0
        mn_error_conn = []
        plt = []
        all_best_list = []
        while max_iter_yc != 0:
            max_iter_yc -= 1
            # 所有种群的误差
            all_error = [0] * len(all_solve_list)
            # 采样个数
            row = math.floor(self.day * 24 * rate)
            for i in range(len(all_error)):
                print("----------cur iter :", max_iter_yc, "total  ",N * 2,"cur i: ", i, "----------")
                # 当前不采样的站点 用二进制表示
                cur_num = all_solve_list[i]
                # cur_num = 68417460044
                print("cur num :",bin(cur_num))
                no_choice_station = self.print_bit(cur_num)
                no_choice_station_num = self.cal_bits_num(cur_num)
                print("no_choice_station_num:", no_choice_station_num)
                no_choice_station_conn = np.zeros([no_choice_station_num, nb_num],dtype=np.int64)
                # 计算每个不采样的点的领居 数量为self.near_s
                for j in range(0, no_choice_station_num):
                    cnt = 0
                    for k in range(0, self.num):
                        if cnt < nb_num :
                            if conn_idx[no_choice_station[j]][k] not in no_choice_station:
                                no_choice_station_conn[j][cnt] = conn_idx[no_choice_station[j]][k]
                                cnt += 1
                print(no_choice_station)
                print("no_choice_station_conn")
                print(no_choice_station_conn)
                # 不采样的点数量加入到其余站点上
                print("原始采样个数: ",row, "采样率: ", row / self.L)
                new_row = row + math.floor((no_choice_station_num * row ) / (self.num - no_choice_station_num))
                if new_row >= self.L :
                    new_row = self.L
                all_sensing_matrix = []
                for station_n in range(self.num):
                    ori_train_data = self.get_train_set(station_n, new_row)
                    dictionary = np.random.normal(loc=0, scale=1 / new_row, size=[new_row, self.L])
                    dictionary = sklearn.preprocessing.normalize(dictionary, norm='l2', axis=0, copy=True,
                                                                 return_norm=False)
                    train_data = ori_train_data.copy()
                    y_learn = train_data.copy()
                    for i_idx in range(max_iter):
                        x = linear_model.orthogonal_mp(dictionary, y_learn, n_nonzero_coefs=n_nonzero)
                        e = np.linalg.norm(y_learn - np.dot(dictionary, x))
                        if e < tolerance:
                            break
                        dict_update(y_learn, dictionary, x, self.L)
                    sensing_matrix = np.dot(dictionary, np.linalg.inv(self.Psi))
                    choiced_sensing_matrix = get_k_col_matrix(sensing_matrix, new_row)
                    all_sensing_matrix.append(choiced_sensing_matrix)
                ori_x = np.zeros([self.num, self.L])
                data_ori = pd.read_csv(self.p)
                for j in range(0, self.num):
                    data_tmp = data_ori.iloc[j * 365 + self.begin_day - self.day : j * 365 + self.begin_day, :]
                    data_tmp_v = data_tmp.values.astype(np.float64)
                    data_tmp_v = my_norm(data_tmp_v)
                    x_ori_tmp = []
                    for idx in range(0, self.day):
                        x_ori_tmp.extend(data_tmp_v[idx])
                    ori_x[j] = x_ori_tmp
                mape_tmp, rmse_tmp, mae_tmp, Y = self.recovery_cal_error(ori_x, all_sensing_matrix, new_row,
                                                                             no_choice_station, no_choice_station_conn)
                all_error[i] = mape_tmp
                if mape_tmp < mn_error :
                    mn_error_num = all_solve_list[i]
                    mn_error = mape_tmp
                    mn_error_conn = no_choice_station_conn.copy()
            print("this iter best:" , mn_error, mn_error_num)
            all_best_list.append(mn_error)
            print("all_best_list", mn_error)
            new_list = [x for _, x in sorted(zip(all_error, all_solve_list))]
            new_list = new_list[0:N]
            new_solve_list = [0] * N
            cross_num = math.floor(N / 2)
            for i in range(cross_num):
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                new_solve_list[i] = self.cross(solve_list[a], solve_list[b], cross_rate)
            for i in range(cross_num, N):
                a = random.randint(0, N - 1)
                new_solve_list[i] = self.variation(solve_list[a], variation_rate)
            plt.append(mn_error)
            print("error list:", plt)
            all_solve_list = new_list + new_solve_list
        print("best:", mn_error,mn_error_num)
        res_list = []
        for i in range(0,self.num):
            if ((mn_error_num >> i) & 1) == 1:
                res_list.append(i)
        filename = './output/npz/iter' + str(max_iter_yc_tmp) + ',n' + str(N) + datetime.datetime.now().strftime('output_%Y%m%d%H%M%S') + '.npz'
        np.savez(filename, A=res_list, B=mn_error_conn, C=plt, D=mn_error_num)
        return res_list, mn_error_conn


    def print_bit(self, a):
        list = [0] * self.num
        no_choic = []
        for i in range(self.num):
            if ((a >> i) & 1) == 1:
                no_choic.append(i)
                list[i] = 1
        return no_choic


    def cal_bits_num(self,a):
        cnt = 0
        for i in range(0, self.num):
            if ((a >> i) & 1) == 1:
                cnt += 1
        return cnt

    def cross(self, a, b, rate):
        len = self.num
        cross_num = math.floor(len * rate)
        remove_list = random.sample(range(len), cross_num)
        remove_list = sorted(remove_list)
        a_list = [0] * len
        for i in range(len):
            if ((a >> i) & 1) == 1:
                a_list[i] = 1

        b_list = [0] * len
        for i in range(len):
            if ((b >> i) & 1) == 1:
                b_list[i] = 1

        new_a_list = []
        for i in range(len):
            if i not in remove_list:
                new_a_list.append(a_list[i])
        new_a_list = new_a_list + b_list[len - cross_num:len]
        res = 0
        for i in range(len):
            if new_a_list[i] == 1:
                res = res | (1 << i)
        return res

    # 变异
    '''
    a 基因a
    len 种群大小
    rate 变异率

    随机打乱a的k个位置
    '''

    def variation(self, a, rate):
        len = self.num
        variation_num = math.floor(len * rate)
        remove_list = random.sample(range(len), variation_num)
        remove_list_idx = remove_list.copy()
        random.shuffle(remove_list_idx)
        a_list = [0] * len
        for i in range(len):
            if ((a >> i) & 1) == 1:
                a_list[i] = 1

        for i in range(variation_num):
            a_list[remove_list[i]] = a_list[remove_list_idx[i]]
        res = 0
        for i in range(len):
            if a_list[i] == 1:
                res = res | (1 << i)
        return res


    def run(self):
        # 字典学习参数
        max_iter = 20
        tolerance = 1e-7
        n_nonzero = 20
        # ----------------#
        res = np.zeros([len(self.rate), 3], dtype=np.float64)
        if is_load == True:
            no_choice_station, no_choice_station_conn = self.load_from_file()
        else :
            no_choice_station, no_choice_station_conn = self.get_no_choice_station()
        # no_choice_station, no_choice_station_conn = self.load_from_file()
        no_choice_station_num = len(no_choice_station)
        for r in range(0, len(self.rate)):
            # 采样个数，同样是测量矩阵的行数
            row = math.floor(self.rate[r] * self.L)
            all_sensing_matrix = []
            all_train_set = []
            new_row = row

            # 是否均分给其他站点
            if self.is_add == True:
                new_row = row + math.floor((no_choice_station_num * row ) / (self.num - no_choice_station_num))

            # 高采样率下会溢出
            if new_row >= self.L :
                new_row = self.L


            # 对于每个站点
            for station_n in range(0, self.num):
                # 获取原始训练集
                # 原始数据集都有，所以一开始每个站点都有独立的测量矩阵。
                # 但是不采样的测量矩阵，没有训练的必要
                # 不采样的站点，测量矩阵放在恢复阶段更新。
                # 跳过 / 不跳过都可
                # if station_n in no_choice_station:
                #     choiced_sensing_matrix = np.zeros([new_row,self.L],dtype=np.int)
                #     all_sensing_matrix.append(choiced_sensing_matrix)
                #     continue
                ori_train_data = self.get_train_set(station_n, new_row)
                all_train_set.append(ori_train_data)
                # print("train set size:",ori_train_data.shape)
                # 获取初始字典
                dictionary = np.random.normal(loc=0, scale=1 / new_row, size=[new_row, self.L])
                dictionary = sklearn.preprocessing.normalize(dictionary, norm='l2', axis=0, copy=True,return_norm=False)
                train_data = ori_train_data.copy()
                # 字典学习

                y_learn = train_data.copy()
                for i in range(max_iter):
                    x = linear_model.orthogonal_mp(dictionary, y_learn, n_nonzero_coefs=n_nonzero)
                    e = np.linalg.norm(y_learn - np.dot(dictionary, x))
                    if e < tolerance:
                        break
                    dict_update(y_learn, dictionary, x, self.L)

                sensing_matrix = np.dot(dictionary, np.linalg.inv(self.Psi))
                # 选取最重要的row列
                print("当前站点:",station_n + 1,"采样个数为：",new_row)
                choiced_sensing_matrix = get_k_col_matrix(sensing_matrix, new_row)
                all_sensing_matrix.append(choiced_sensing_matrix)
            # print("size",len(all_sensing_matrix),len(all_train_set))
            # print(all_sensing_matrix)
            data_ori = pd.read_csv(self.p)
            mapeave = 0
            rmseave = 0
            maeave = 0
            for _time in range(0, self.time):
                print("NO-------------------------------------------------------------", _time + 1)
                # 截取每个站点当前需要测试的序列
                # 即使不采也要计算误差，所以全加入。
                ori_x = np.zeros([self.num, self.L])

                for j in range(0, self.num):
                    data_tmp = data_ori.iloc[(j * 365) + self.begin_day + (_time * self.day):(j * 365) + self.begin_day + (_time + 1) * self.day, :]
                    print("当前站点", j + 1, "第", _time + 1, "轮采样","采样个数为:",new_row, "采样率:",self.rate[r])
                    print("采样起始时间是", self.begin_day, "; 每次采样", self.day, "天")
                    print("test data from", (j * 365) + self.begin_day + (_time * self.day), "to",(j * 365) + self.begin_day + (_time + 1) * self.day)
                    data_tmp_v = data_tmp.values.astype(np.float64)
                    data_tmp_v = my_norm(data_tmp_v)
                    x_ori_tmp = []
                    for idx in range(0, self.day):
                        x_ori_tmp.extend(data_tmp_v[idx])
                    ori_x[j] = x_ori_tmp
                # print(ori_x)
                mape_tmp, rmse_tmp, mae_tmp,Y = self.recovery_cal_error(ori_x, all_sensing_matrix, new_row,no_choice_station,no_choice_station_conn)
                mapeave += mape_tmp
                rmseave += rmse_tmp
                maeave += mae_tmp
                # 在线更新
                Y = np.reshape(Y, (self.num, new_row))
                # Y = Y.T
                # print("sensing Y size",Y.shape)
                for _idx in range(0, self.num):
                    # 不采的同样不更新
                    if _idx in no_choice_station:
                        continue
                    # 更新训练集
                    all_train_set[_idx] = np.insert(all_train_set[_idx][:, 1:], self.datasize - 1, values=Y[_idx],axis=1)
                    # 更新初始字典
                    dictionary = np.random.normal(loc=0, scale=1 / new_row, size=[new_row, self.L])
                    dictionary = sklearn.preprocessing.normalize(dictionary, norm='l2', axis=0, copy=True,return_norm=False)
                    # 字典学习
                    y_learn = all_train_set[_idx].copy()
                    for i in range(max_iter):
                        x = linear_model.orthogonal_mp(dictionary, y_learn, n_nonzero_coefs=n_nonzero)
                        e = np.linalg.norm(y_learn - np.dot(dictionary, x))
                        if e < tolerance:
                            break
                        dict_update(y_learn, dictionary, x, self.L)

                    sensing_matrix = np.dot(dictionary, np.linalg.inv(self.Psi))
                    # 选取最重要的row列
                    # print("current station:", _idx + 1)
                    choiced_sensing_matrix = get_k_col_matrix(sensing_matrix, new_row)
                    all_sensing_matrix[_idx] = choiced_sensing_matrix

            mapeave /= self.time
            rmseave /= self.time
            maeave /= self.time
            res[r][0] = mapeave
            res[r][1] = rmseave
            res[r][2] = maeave
        return res

    def get_y_time_conn(self,ori_fi,m):
        # 计算每个站点采样的时刻与之对应的y元素位置.
        # 简单来说就是计算每行的1在第几列
        # 如：
        # fi = 0100000
        #      1000000
        #      0001000
        # 结果就是 103
        #
        res = []
        for i in range(self.num) :
            # tmp = np.zeros(self.L,dtype=np.int)
            tmp = np.full((self.L),-1)
            for j in range(m):
                for k in range(self.L):
                    if ori_fi[i][j][k] == 1:
                        tmp[k] = j
            res.append(tmp)

        return res


    # 恢复并且计算误差
    def recovery_cal_error(self, ori_x, ori_fi, m,no_choice_station,no_choice_station_conn):
        '''
        论文出处：Joint Dictionary Learning and Recovery Algorithms in a Jointly Sparse Framework
        ori_x : 原始数据，待采集的，维度应当是 L*n,L为站点个数，n为单条序列长度
        ori_fi : 应当是 L 个 m * n的矩阵拼起来的，形如 行数为m，列数为L*n
        '''
        # 首先要计算每个站点他的y所对应的是第几个时间段。
        # print(no_choice_station)
        # print(no_choice_station_conn)
        print("采样的站点单站点采样个数： ", m)
        y_time_conn = self.get_y_time_conn(ori_fi,m)
        # 预处理，不采样的站点利用conn站点权重注入
        # 原始所有站点Y
        Y_all = []
        for i in range(0,self.num):
            tmp  = ori_x[i].T.copy()
            Y_all.append(np.dot(ori_fi[i], tmp))
        # 对于每个不采样的站点来说
        for i in range(0,len(no_choice_station)) :
            # 计数器
            # 第一维计算每个时间点的采样个数之和
            # 第二维记录是那几个站点采样了该时间点
            cnt_matrix = np.zeros([2,self.L],dtype=np.int64)
            for j in range(0,self.near_s) :
                tmp = y_time_conn[no_choice_station_conn[i][j]]
                for k in range(len(tmp)):
                    if tmp[k] != -1:
                        # 位运算记录那几个站点贡献
                        cnt_matrix[0][k] += 1
                        cnt_matrix[1][k] = cnt_matrix[1][k] | (1 << no_choice_station_conn[i][j])
            # 统计各个时刻三站点的情况
            three = [x for x in range(self.L) if cnt_matrix[0][x] == 3]
            two = [x for x in range(self.L) if cnt_matrix[0][x] == 2]
            one = [x for x in range(self.L) if cnt_matrix[0][x] == 1]
            # 全部打乱 然后连起来
            random.shuffle(one)
            random.shuffle(two)
            random.shuffle(three)
            # 最终对于该站点要采集的点集合final_choice
            final_choice = three + two + one
            # 由于最前面是three ,two所以截取前m个元素 存储该不采样站点要注入的时刻
            final_choice = final_choice[0:m]
            # 根据集合构建测量矩阵
            final_sensing_matrix = np.zeros([m,self.L],dtype=np.int64)
            cur = 0
            # 加权得到采样y
            final_y = []
            for k in range(len(final_choice)):
                final_sensing_matrix[cur][final_choice[k]] = 1
                cur += 1
                # 更新采样y  由相邻的加权得到。
                y_tmp = 0
                s_cnt = 0
                # 位运算的保存值。
                cnt_tmp = cnt_matrix[1][final_choice[k]]
                for j in range(self.num):
                    # 如果是该站点贡献的
                    if ((cnt_tmp >> j) & 1) == 1 :
                        # Y_all 保存所有站点的采样值，若是第j站点贡献，第一维自然是j
                        # 第二维解释： y_time_conn 存的是第 i 时间点采集 -> y 第 y_time_conn[j][i] 个元素的映射 于是是final_choice[i]
                        y_tmp += Y_all[j][y_time_conn[j][final_choice[k]]]
                        s_cnt += 1
                # 此处的加权仅仅取了平均
                if s_cnt != 0:
                    y_tmp /= s_cnt
                else :
                    print("s_cnt == 0")
                y_tmp = round(y_tmp,5)
                final_y.append(y_tmp)
            # 更新测量矩阵
            ori_fi[no_choice_station[i]] = final_sensing_matrix
            Y_all[no_choice_station[i]] = final_y
        for i in range(0,36):
            if i not in no_choice_station:
                print("第",i,"个站点采样时刻：")
                sample_time = []
                sensing_t = ori_fi[i]
                for j in range(0,sensing_t.shape[0]) :
                    for k in range(0, sensing_t.shape[1]):
                        if sensing_t[j][k] == 1:
                            sample_time.append(k)
                print(sample_time)

        # 采样矩阵Y
        Y = []
        for i in range(len(Y_all)) :
            Y.extend(Y_all[i])
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
        print("begin recovery")
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
        no_sample_cnt = 0
        for i in range(0, self.num):
            # 不采的站点用加权和替代
            if i in no_choice_station :
                for j in range(0,m):
                    for k in range(0,self.L):
                        if ori_fi[i][j][k] == 1:
                            res_x[i][k] = Y_all[i][j]
                            no_sample_cnt += 1
                            break
            else :
                # 采的站点用原始数据替代
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
        print("结果矩阵size:",res_x.shape)
        for i in range(0, self.num):
            for j in range(0, self.L):
                if ori_x[i][j] == 0:
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
        print("不采样的站点个数为: ", len(no_choice_station))
        print("当前采样个数：",cntt,"实际计算的个数", cnt, "缺失率：", round(100 - cntt / cnt * 100,3),"mape为：",mape,"  rmse 为:",rmse,"  mae为:",mae)
        print("替换了： ", no_sample_cnt)
        ## 画图
        # print(res_x)
        # print(ori_x)
        # x_line = np.linspace(0, ori_x.shape[1] - 1, ori_x.shape[1])
        # for pltidx in range(0,ori_x.shape[0]) :
        #     fig = plt.figure(figsize=(12, 14), dpi=300)
        #     plt.title("Sample rate : " + str(m / self.L) + "    station: " + str(pltidx))
        #     plt.plot(x_line, ori_x[pltidx], c='black', ls='-', alpha=1, label='ori_x')
        #     plt.plot(x_line, res_x[pltidx], c='red', ls=':', alpha=1, label='res_x')
        #     plt.legend()
        #     plt.show()
        return mape, rmse,mae, Y

    # 初始训练集为随机选取
    def get_train_set(self, station, row):
        # 此处 例如self.begin_day 从200 天开始
        # 单条为 5 天 ， 总共 5 条
        # 则训练集 是从 200 - 5 * 5 开始
        begin_day = self.begin_day - self.day * self.datasize
        data = pd.read_csv(self.p)
        data = data.iloc[(station * 365) + begin_day: (station * 365) + self.begin_day, :]
        # print("train from", (station * 365) + begin_day, "to", (station * 365) + self.begin_day)
        data = data.values.astype(np.float64)
        data = my_norm(data).copy()
        # data 的列数为24  需要按行排放成 datasize , (day * 24) 的矩阵
        res_data = np.zeros([self.datasize, self.L])
        for i in range(self.datasize):
            tmp = []
            idx = 0
            for _ in range(self.day):
                tmp.extend(data[idx])
                idx += 1
            res_data[i] = tmp
        # 对每一行进行随机抽取 row 元素组成新矩阵
        # 保持相对顺序不变
        new_res = np.zeros([self.datasize, row])
        for i in range(self.datasize):
            rand = np.zeros(self.L)
            for idx in range(row):
                rand[idx] = 1
            # 打乱，为 1 代表选中
            np.random.shuffle(rand)
            idx = 0
            for k in range(self.L):
                if rand[k] == 1:
                    new_res[i][idx] = res_data[i][k]
                    idx += 1
        new_res = new_res.T
        return new_res


def my_norm(data):
    return data


def dict_update(y, d, x, n_components):
    for i in range(n_components):
        index = np.nonzero(x[i, :])[0]
        if len(index) == 0:
            continue
        d[:, i] = 0
        r = (y - np.dot(d, x))[:, index]
        u, s, v = np.linalg.svd(r, full_matrices=False)
        d[:, i] = u[:, 0]
        for j, k in enumerate(index):
            x[i, k] = s[0] * v[0, j]
    return d, x


# 三对角矩阵 稀疏基
def trip1(m):
    Psi = np.eye(m)
    for i in range(m):
        if i < m - 1 and i > 0:
            Psi[i][i + 1] = -1
            Psi[i][i - 1] = -1
        elif i == 0:
            Psi[i][i + 1] = -1
        elif i == m - 1:
            Psi[i][i - 1] = -1
        Psi[i][i] = 2
    return np.linalg.inv(Psi)


def trip2(m):
    Psi = np.eye(m)
    for i in range(m):
        if i < m - 1:
            Psi[i][i+1] = -1
    return np.linalg.inv(Psi)
