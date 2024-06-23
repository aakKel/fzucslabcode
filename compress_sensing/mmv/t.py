import numpy as np
from multi_cssp import multi_cssp
from mutilcsspv2 import multi_cssp2
from multi_csspv3 import multi_cssp3
from multi_csspv4 import multi_cssp4
from multi_random import multi_random
from multi_greedy import multi_greedy
from multi_epprandom import multi_epprandom
from doubleGa import multi_cssp5
from GABaseline import GABaseline
import pandas as pd
import sys
import datetime
import time
import logging
# start_time = time.clock()
# now = datetime.datetime.now()
# filename = now.strftime('./output/output_%Y%m%d%H%M%S.txt')
# sys.stdout = open(filename, 'w')
# p为mean后的，取24小时均值
# zero_p 为 原始有缺失的
# zero_p = '../dataset/AQI_beijing.csv'
# p = '../dataset/AQI_beijing_mean.csv'
# p = '../dataset/shanghai2014_mean_single_station.csv'
# zero_p = '../dataset/shanghai2014_0_single_station.csv'

p = '../dataset/36/pm2.5_bj_20140501-20150430_mean.csv'
zero_p = '../dataset/36/pm2.5_bj_20140501-20150430_0.csv'
# 1 - 365 第一站点 以此类推


# 采样率
# rate = [0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17]
rate = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
# rate = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
# 站点个数
# num =
# 单条向量长度为 day * 24
# day =
# 训练集序列个数
# datasize =
# 采样几次
# time_t =
# 从第几天开始采，此处应注意 需要大于训练集的大小
# begin_day =
# 重复实验取平均值
# iter_time =
result = np.zeros([len(rate), 3], dtype=np.float64)
for _i in range(0, iter_time):
    print("重复实验，第", _i)
    # tmp = multi_cssp(num,begin_day,day,datasize,time_t,rate,p,zero_p).run()
    # tmp = multi_cssp4(num, begin_day, day, datasize, time_t, rate, p, zero_p).run()
    tmp = GABaseline(num, begin_day, day, datasize, time_t, rate, p, zero_p).run()
    # tmp = multi_cssp5(num, begin_day, day, datasize, time_t, rate, p, zero_p).run()
    # tmp = multi_random(num, begin_day, day, datasize, time_t, rate, p, zero_p).run()
    # tmp = multi_greedy(num, begin_day, day, datasize, time_t, rate, p, zero_p).run()
    # tmp = multi_epprandom(num, begin_day, day, datasize, time_t, rate, p, zero_p).run()
    result = np.add(result,tmp)

result = result / iter_time
for i in range(0,len(rate)) :
    # print("采样率：",rate[i] * 100,"%,MAPE为",result[i][0],"RMSE为",result[i][1])
    print(str(rate[i] * 100)+"%", round(result[i][0],3), round(result[i][1],3),round(result[i][2],3))
# print(result)


# end_time = time.clock()    # 程序结束时间
# run_time = end_time - start_time    # 程序的运行时间，单位为秒
# print("run_time", run_time)
