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
from pack import trip2, recovery_cal_error, trip1
import datetime
import time
is_load_from_file = True
all_file = []
class GABaseline:
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
        self.is_add = True
        self.solve_num = 0

    def load(self, row, iter_time):
        file_10 = './output/npz/03-18-23-19/row_16, iter_1000, solve_num_10.npz'
        file_15 = './output/npz/03-19-14-19/row_25, iter_1000, solve_num_10.npz'
        file_20 = './output/npz/03-20-14-00/row_33, iter_1000, solve_num_10.npz'
        file_25 = './output/npz/03-22-20-41/row_42, iter_1000, solve_num_10.npz'
        file_30 = './output/npz/03-23-01-05/row_50, iter_1000, solve_num_10.npz'
        file_35 = './output/npz/03-23-06-42/row_58, iter_1000, solve_num_10.npz'
        file_40 = './output/npz/03-23-12-01/row_67, iter_1000, solve_num_10.npz'
        file_45 = './output/npz/03-27-13-39/row_75, iter_1000, solve_num_10.npz'
        all_file = [file_10, file_15, file_20, file_25, file_30, file_35, file_40, file_45]

        cur_file = all_file[iter_time]
        data = np.load(cur_file)
        A = data['A'].tolist()
        sensing = []
        for i in range(self.num):
            cur_sensing = np.zeros([row, self.day * self.datalen])
            idx = 0
            for j in range(row):
                for k in range(self.day * self.datalen):
                    if A[i][j][k] == 1:
                        cur_sensing[idx][k] = 1
                        idx += 1
                        break
            sensing.append(cur_sensing)
        return sensing


    def run(self):
        res = np.zeros([len(self.rate), 3], dtype=np.float64)
        start_time = time.time()
        for r in range(0,len(self.rate)):
            # 采样个数，同样是测量矩阵的行数
            row = math.floor(self.rate[r] * self.L)
            if is_load_from_file == True:
                sensing_matrix = self.load(row, r)
            else:
                sensing_matrix = self.get_sensing_matrix(row, start_time)
            data_ori = pd.read_csv(self.p)
            mapeave = 0
            rmseave = 0
            maeave = 0
            for _time in range(0, self.time):
                print("NO-------------------------------------------------------------", _time + 1)
                # 截取每个站点当前需要测试的序列
                ori_x = np.zeros([self.num, self.L])
                for j in range(0, self.num):
                    data_tmp = data_ori.iloc[(j * 365) + self.begin_day + (_time * self.day):(j * 365) + self.begin_day + (
                                _time + 1) * self.day, :]
                    print("当前站点", j + 1, "第", _time + 1, "轮采样", "采样个数为:", row, "采样率为:", self.rate[r])
                    print("采样起始时间是", self.begin_day, "; 每次采样", self.day, "天")
                    print("test data from", (j * 365) + self.begin_day + (_time * self.day), "to",
                          (j * 365) + self.begin_day + (_time + 1) * self.day)
                    data_tmp_v = data_tmp.values.astype(np.float64)
                    # data_tmp_v = my_norm(data_tmp_v)
                    x_ori_tmp = []
                    for idx in range(0, self.day):
                        x_ori_tmp.extend(data_tmp_v[idx])
                    ori_x[j] = x_ori_tmp
                # print(ori_x)
                mape_tmp, rmse_tmp, mae_tmp, Y = recovery_cal_error(self, ori_x, sensing_matrix, row)
                mapeave += mape_tmp
                rmseave += rmse_tmp
                maeave += mae_tmp

            mapeave /= self.time
            rmseave /= self.time
            maeave /= self.time
            res[r][0] = mapeave
            res[r][1] = rmseave
            res[r][2] = maeave

        return res




    def get_sensing_matrix(self, sample_num, start_time):
        sample_all_len = 24 * self.day
        iter_time = 1000
        # 染色体个数
        solve_num = 10
        same_cnt_max = 20
        self.solve_num = solve_num
        all_solve = []
        # 初始种群 大小为 站点个数 × 采样总长度， 每一行1的个数为需要采样的个数，即为sample_num
        for idx in range(solve_num):
            solve_tmp = np.zeros((self.num, sample_all_len))
            for i in range(self.num):
                ones_indices = np.random.choice(sample_all_len, sample_num, replace=False)
                solve_tmp[i, ones_indices] = 1
            all_solve.append(solve_tmp)
        # 迭代

        mn_error_all = []
        before_solve = []
        same_cnt = 0
        for iter_idx in range(iter_time):
            print("当前迭代次数：", iter_idx)
            cur_time = time.time()
            run_time = cur_time - start_time
            print("已运行时间: {:.2f}秒".format(run_time))
            # 当前迭代中的每个解的误差
            solve_error = []
            for s_idx in range(solve_num):
                cur_solve = all_solve[s_idx]
                # 当前解所有的测量矩阵 维度为站点个数 × 采样个数 × 总采样长度
                all_sensing_matrix = []
                for i in range(cur_solve.shape[0]):
                    cur_sensing = np.zeros([sample_num, sample_all_len])
                    cur_idx = 0
                    for j in range(cur_solve.shape[1]):
                        if cur_solve[i][j] == 1:
                            cur_sensing[cur_idx][j] = 1
                            cur_idx += 1
                    all_sensing_matrix.append(cur_sensing)
                # print(all_sensing_matrix)

                # 训练集
                ori_x = np.zeros([self.num, self.L])
                data_ori = pd.read_csv(self.p)
                for j in range(0, self.num):
                    data_tmp = data_ori.iloc[j * 365 + self.begin_day - self.day: j * 365 + self.begin_day, :]
                    data_tmp_v = data_tmp.values.astype(np.float64)
                    # data_tmp_v = my_norm(data_tmp_v)
                    x_ori_tmp = []
                    for idx in range(0, self.day):
                        x_ori_tmp.extend(data_tmp_v[idx])
                    ori_x[j] = x_ori_tmp
                mape_tmp, rmse_tmp, mae_tmp, Y = recovery_cal_error(self, ori_x, all_sensing_matrix, sample_num)
                solve_error.append(mape_tmp)
            print("当前解的误差list :", solve_error)
            # 最好的下标
            sorted_indices = [index for index, value in sorted(enumerate(solve_error), key=lambda x: x[1])]
            mn_error_all.append(all_solve[sorted_indices[0]])
            next_solve = []
            print("选取一半作为下一组解：", end="")
            for i in range((int)(len(sorted_indices)/2)):
                next_solve.append(all_solve[sorted_indices[i]])
                print(solve_error[sorted_indices[i]], end=" ")
            print()
            left_num = len(sorted_indices) - len(next_solve)
            numbers = list(range(solve_num))
            # 随机选择两个不相等的整数
            random_numbers = random.sample(numbers, 2)
            for i in range(left_num):
                next_solve.append(self.get_solve(all_solve, random_numbers[0], random_numbers[1], sample_num))
            all_solve = next_solve.copy()
            self.print_all_sample(next_solve.copy())

            if len(before_solve) == 0:
                before_solve = next_solve.copy()
                continue

            if self.check_equ(before_solve, next_solve):
                same_cnt += 1
            else:
                before_solve = next_solve.copy()
                same_cnt = 1
            if same_cnt > same_cnt_max:
                break
        res_sensing_matrix = []
        cur_solve = all_solve[0]
        # 当前解所有的测量矩阵 维度为站点个数 × 采样个数 × 总采样长度
        for i in range(cur_solve.shape[0]):
            cur_sensing = np.zeros([sample_num, sample_all_len])
            cur_idx = 0
            for j in range(cur_solve.shape[1]):
                if cur_solve[i][j] == 1:
                    cur_sensing[cur_idx][j] = 1
                    cur_idx += 1
            res_sensing_matrix.append(cur_sensing)
        cur_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
        folder_path = './output/npz/' + cur_time

        os.makedirs(folder_path)
        filename = folder_path + '/row_' + str(sample_num) + ', iter_' + str(iter_time) + ', solve_num_' + str(solve_num) + '.npz'
        # filename = './output/npz/gabaseline_iter' + str(iter_time) + ',n' + str(solve_num) + datetime.datetime.now().strftime('output_%Y%m%d%H%M%S') + '.npz'
        np.savez(filename, A=res_sensing_matrix, B=mn_error_all)
        return res_sensing_matrix

    def check_equ(self, before_solve, next_solve):
        for i in range(len(before_solve)):
            for j in range(len(before_solve[i])):
                for k in range(len(before_solve[i][j])):
                    if (before_solve[i][j][k] != next_solve[i][j][k]):
                        return False

        return True

    def print_all_sample(self, all_solve):
        for i in range(self.solve_num):
            print("第", i, "个解：", end=" ")
            cur_solve = all_solve[i]
            for j in range(cur_solve.shape[0]):
                print("站点", j, "采样时刻：", end=" ")
                for k in range(cur_solve.shape[1]):
                    if cur_solve[j][k] == 1:
                        print(k, end=" ")
                print()


    def get_solve(self, all_solve, idx_i, idx_j, sample_num):
        solve1 = all_solve[idx_i].copy()
        solve2 = all_solve[idx_j].copy()
        result = np.multiply(solve1, solve2)
        has_vec = []
        # 记录S1 s2 采样的时刻
        for i in range(result.shape[0]):
            cur_has_vec = []
            for j in range(result.shape[1]):
                if (solve1[i][j] == 1) and (solve2[i][j] == 1):
                    continue
                if (solve1[i][j] == 1) or (solve2[i][j] == 1):
                    cur_has_vec.append(j)
            has_vec.append(cur_has_vec)

        for i in range(result.shape[0]):
            cnt = 0
            sampled_idx = []
            for j in range(result.shape[1]):
                if result[i][j] == 1:
                    cnt += 1
                    sampled_idx.append(j)
            left_num = sample_num - cnt
            choice_list = has_vec[i]
            random_positions = random.sample(range(len(choice_list)), left_num)
            for j in range(len(random_positions)):
                result[i][choice_list[random_positions[j]]] = 1
        # check error
        for i in range(result.shape[0]):
            cntt = 0
            for j in range(result.shape[1]):
                if result[i][j] == 1:
                    cntt += 1
            if cntt != sample_num:
                print(i, "error for get solve", cntt, sample_num)
        return result

# 恢复并且计算误差
    def recovery_cal_error(self,ori_x,ori_fi,m):
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
        for i in range(0,self.num):
            A[i] = np.dot(A[i],self.Psi)
        # 把num个A拼成斜对角矩阵
        fi = np.block([[A[i] if i == j else np.zeros((m, self.L)) for j in range(len(A))] for i in range(len(A))])
        xk = np.dot(np.linalg.pinv(fi), Y)
        #----------------------------------#
        tiny_error = 0.9  # 惩罚项      参数
        p = 0.1  # 0<= p <=1           参数
        min_error = 2  # 终止的误差限   参数
        LD = 0.2  # 考虑噪声误差         参数
        #----------------------------------#
        ## 算法流程见论文
        iter_cnt = 0
        while 1:
            iter_cnt += 1
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
        print("恢复算法迭代次数: ",iter_cnt)
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
                    continue
                cnt += 1
                mape += np.abs(ori_x[i][j] - res_x[i][j]) / ori_x[i][j]
                rmse += (ori_x[i][j] - res_x[i][j]) ** 2
                mae += np.abs(ori_x[i][j] - res_x[i][j])

        mape = (mape / cnt) * 100
        mape = round(mape,3)
        rmse = rmse / cnt
        rmse = math.sqrt(rmse)
        rmse = round(rmse, 3)
        mae = mae/cnt
        mae = round(mae, 3)

        if mape > 100:
            mape = 100
        print("当前采样个数：",cntt,"实际计算的个数", cnt, "缺失率：", round(100 - cntt / cnt * 100,3),"mape为：",mape,"  rmse 为:",rmse,"  mae为:",mae)

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
        return mape,rmse,mae,Y










