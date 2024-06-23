import math

import matplotlib.pyplot as plt
import random
import pandas as pd

import numpy as np


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

def multi_weber(points, n_facilities, max_iter, pop_size, mutation_rate):
    """
    多远韦伯问题求解函数
    :param points: 全部点的坐标，包括候选设施和需求点
    :param n_facilities: 设施数量
    :param max_iter: 最大迭代次数
    :param pop_size: 种群大小
    :param mutation_rate: 变异率
    :return: 最优设施位置列表，最小距离和
    """
    # 初始化种群
    pop = []
    for i in range(pop_size):
        pop.append(random.sample(points, n_facilities))
    # 初始化适应度函数值
    fits = [0] * pop_size
    # 迭代搜索
    res = []
    res_fit = 1e6
    for i in range(max_iter):
        # fits = [0] * pop_size
        # 计算适应度函数值
        for j in range(pop_size):
            fit = 0
            for p in points:
                # d = min([np.linalg.norm(np.array(p) - np.array(f)) for f in pop[j]])
                # d = min([distance(p[0], p[1], f[0], f[1]) for f in pop[j]])
                # fit += d
                # 跳过自身
                d = []
                for f in pop[j]:
                    if f == p:
                        continue
                    d.append(distance(p[0],p[1],f[0],f[1]))
                fit += min(d)
            fits[j] = fit
        # 选择操作
        new_pop = []
        fits = np.array(fits)
        for j in range(pop_size):
            idx = np.random.choice(range(pop_size), size = 2, replace=False)
            if fits[idx[0]] < fits[idx[1]]:
                new_pop.append(pop[idx[0]])
            else:
                new_pop.append(pop[idx[1]])
        # 变异操作
        for j in range(n_facilities):
            if random.random() < mutation_rate:
                idx = random.randint(0, n_facilities - 1)
                pop[j][idx] = random.choice(points)
        # 交叉操作
        for j in range(int(pop_size / 2)):
            idx = random.sample(range(n_facilities), 2)
            new_pop[j] = pop[j][:idx[0]] + pop[j + 1][idx[0]:idx[1]] + pop[j][idx[1]:]
            new_pop[j + int(pop_size / 2)] = pop[j + 1][:idx[0]] + pop[j][idx[0]:idx[1]] + pop[j + 1][idx[1]:]
        pop = new_pop
        # 返回最优解

        best_fit = min(fits)
        best_idx = np.argmin(fits)
        ff = list(set(pop[best_idx]))
        if len(ff) > 20:
            continue

        if best_fit < res_fit :
            res_fit = best_fit
            res = pop[best_idx]
    return res,res_fit

def plot_multi_weber(points, facilities):
    """
    将多远韦伯问题的最优解用图形表示
    :param points: 全部点的坐标，包括候选设施和需求点
    :param facilities: 最优设施位置列表
    """
    fig = plt.figure(figsize=[14,14],dpi = 300)
    # 绘制需求点
    demand_points = [p for p in points if p not in facilities]
    plt.scatter([p[0] for p in demand_points], [p[1] for p in demand_points], s=50, c='r', marker='o')
    # 绘制候选设施
    # candidate_facilities = [p for p in points if p in facilities]
    # plt.scatter([p[0] for p in candidate_facilities], [p[1] for p in candidate_facilities], s=100, c='green', marker='s')
    # 绘制最优设施位置
    plt.scatter([p[0] for p in facilities], [p[1] for p in facilities], s=150, c='green', marker='^')
    # 绘制需求点到最近设施之间的连线
    for p in demand_points:
        distances = [distance(p[0], p[1], f[0], f[1]) for f in facilities]
        idx = distances.index(min(distances))
        plt.plot([p[0], facilities[idx][0]], [p[1], facilities[idx][1]], linestyle='--', color='gray')
    plt.savefig('./output/ycsf.svg')
    plt.show()

p = '../dataset/36/pm25_latlng.csv'
data = pd.read_csv(p)
data = data.iloc[:, 1:]
data = data.values.astype(np.float64)

data_list = data.tolist()

points = [(row[0], row[1]) for row in data_list] # 原始经纬度信息
#points, n_facilities, max_iter, pop_size, mutation_rate
facilities,sum = multi_weber(points,15,100,100,0.25)
facilities = list(set(facilities))
print(facilities)
print(len(facilities))
plot_multi_weber(points, facilities)
print(sum)