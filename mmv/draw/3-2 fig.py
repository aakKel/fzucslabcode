import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
p = '../../dataset/36/pm2.5_bj_20140501-20150430_mean.csv'
size = 5
begin = 124
plt.rcParams['font.sans-serif'] = ['Times New Roman']
data_ori = pd.read_csv(p)
data_tmp = data_ori.iloc[begin:size + begin:]
data_tmp_v = data_tmp.values.astype(np.float64)
y = []
x = list(range(1, size * 24 + 1))
for i in range(0, data_tmp_v.shape[0]) :
    y.extend(data_tmp_v[i])
# y = y / np.linalg.norm(y)
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
# plt.title('折线图示例')
plt.xlabel('Time/h',fontsize=15)
plt.ylabel('PM2.5', fontsize=15)
plt.xticks(fontsize=14)  # 设置x轴刻度标签字体大小
plt.yticks(fontsize=14)
plt.plot(x, y, linestyle='-', color='black')
# plt.show()
plt.savefig('3.3.4norm-tmp-af.svg')
