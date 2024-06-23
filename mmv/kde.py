import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
# 生成一些随机的二维数据点
p = '../dataset/36/pm25_latlng.csv'
data = pd.read_csv(p)
data = data.iloc[:, 1:]
data = data.values.astype(np.float64)
print(data)
# 计算核密度估计
kde = gaussian_kde(data.T)
# 生成一个网格，用于绘制等高线图
x, y = np.mgrid[39.5:40.52:1000j, 115.9:117.2:1000j]
positions = np.vstack([x.ravel(), y.ravel()])
z = np.reshape(kde(positions).T, x.shape)
fig, ax = plt.subplots(figsize=[14,14])
ax.contourf(x, y, z, cmap='Blues')
for i in range(len(data)):
    ax.scatter(data[i][0], data[i][1],s=10)
    ax.annotate(str(i), (data[i][0], data[i][1]))
# 绘制等高线图
fig.set_dpi(300)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.rcParams['font.size'] = 14
plt.savefig('kde.svg')
plt.show()

