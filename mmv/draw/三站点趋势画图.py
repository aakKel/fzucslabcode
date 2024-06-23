import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 防止乱码
# plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['font.sans-serif'] = ['Times New Roman']

plt.rcParams['axes.unicode_minus'] = False

p = '../dataset/shanghai2014_mean_single_station.csv'

station = 1
day = 7
datasize = 4
time = 8
# beginday = 200 - day * datasize
beginday = 30 - day * datasize
fig = plt.figure(figsize=(12, 14), dpi=300)
# plt.title('2月-4月')
for idx in range(3):
    data = pd.read_csv(p)
    # data = data.iloc[beginday:beginday + (datasize + time) * day ,:]
    data = data.iloc[idx * 365 + beginday:idx * 365+ beginday + (datasize + time) * day ,:]
    data = data.values.astype(np.float64)
    ori_x = []
    for i in range(0,data.shape[0]):
        ori_x.extend(data[i])
    print(ori_x)

    print(data.shape)
    l = len(ori_x)
    print(l)
    x = np.linspace(0,2016,l)
    x1 = x[0:672]
    y1 = ori_x[0:672]
    plt.subplot(3,1,idx + 1)
    x2 = x[672:2016]
    y2 = ori_x[672:2016]
    # fig = plt.figure(figsize=(12, 4), dpi=600)
    # plt.title('station '+str(idx + 1))
    plt.xlabel('Time/hour')
    plt.ylabel('PM2.5')
    plt.plot(x1,y1,c='black', ls='-',alpha=1, label='Train set')
    plt.plot(x2,y2,c='black', ls=':',alpha=1, label='Test set')
    plt.legend(loc=0)
# plt.savefig("2-4.svg")
plt.show()

