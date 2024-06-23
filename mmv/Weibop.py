'''
1. 初始化n个种群 作为原始祖先解

2. 交叉变异 生成x + y = n个解

3. x : n个祖先 随机互换  交叉：

a：0b 011100101
b: 0b 100101001

随机选取a的3个 剔除， 将b低3位插入在a后面

4. y : 打乱排序

5. 计算2n个基因的适应度（恢复误差） 将误差最低的n个作为原始祖贤解

'''
import math
import random
import pandas as pd
import numpy as np

from pearsonr import get_pearsonr

# 交叉
'''
a 基因a
len 种群大小
rate 变异率

具体为a随机选取 k 个不要的位置，移除，将b后k位插入a中
'''
def cross(a, b, len, rate) :
    cross_num = math.floor(len * rate)
    remove_list = random.sample(range(len),cross_num)
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
    print(new_a_list)
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
def variation(a, len, rate):
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
    print(a_list)
    for i in range(len):
        if a_list[i] == 1:
            res = res | (1 << i)
    return res


a = 1010
print(bin(a))
b = 1212
print(bin(b))
c = cross(a,b,36,0.3)
print(bin(c))

c = variation(c,36,0.3)
print(bin(c))





# p = '../dataset/36/pm2.5_bj_20140501-20150430_mean.csv'
# station_num = 36
# begin = 100
# day = 7
# data = pd.read_csv(p)
# a = range(0,station_num)
# data_ori = np.zeros([len(a),day * 24])
# res, indices = get_pearsonr(p,day,begin,station_num,False)
# for i in range(0,len(a)):
#     data_tmp = data.iloc[(a[i] * 365) + begin :(a[i] * 365) + begin + day, :]
#     data_tmp = data_tmp.values.astype(np.float64)
#     y = []
#     for j in range(day) :
#         y.extend(data_tmp[j])
#     data_ori[i] = y




