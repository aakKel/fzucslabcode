import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# X轴数据
# x = [1, 2, 3, 4, 5]
#
# # Y轴数据
# y = [10, 8, 6, 4, 2]
#
# # 绘制折线图
# plt.plot(x, y, marker='o', linestyle='-', color='blue')
#
# # 添加标题和标签
# plt.title("折线图和散点图")
# plt.xlabel("sample rate")
# plt.ylabel("mape")
#
# # 创建新的坐标系
# ax = plt.gca().twinx()
#
# # 绘制散点图
# x_scatter = [1, 3, 5]
# y_scatter = [8, 4, 2]
# ax.scatter(x_scatter, y_scatter, marker='o', color='red')
#
# # 设置散点图的Y轴范围
# ax.set_ylim([0, 10])
#
# # 显示图形
# plt.show()


# #a = 7 #
# choice = [6, 8, 9, 11, 14, 19, 21]
# y1 = [68.969,36.604,23.898,16.572,11.201,7.499]

# a = 13
choice =[1, 2, 3, 6, 7, 8, 10, 12, 13, 14, 17, 19, 21]
y1=[55.837,32.315,18.555,11.296,6.007,1.916,]

x1 = [10,20,30,40,50,60]
y2 = [76.272,41.59,28.634,20.848,15.587,11.59,]
p = '../dataset/36/pm25_latlng.csv'
data = pd.read_csv(p)
data = data.iloc[:, 1:]
data = data.values.astype(np.float64)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),dpi = 300)
ax1.plot(x1,y1,color='black',ls=':',label='a = 3')
ax1.plot(x1,y2,color='black',ls='-',label='Ori')
ax1.set_xlabel('sample rate(%)')
ax1.set_ylabel('MAPE')
ax1.legend()

f = False
for i in range(36):
    if i in choice:
        if f == False:
            ax2.scatter(data[i][0],data[i][1],color='red',label='choice')
            f = True
        else :
            ax2.scatter(data[i][0], data[i][1], color='red')
    else :
        ax2.scatter(data[i][0], data[i][1], color='black')
ax2.legend()
ax2.set_xlabel('latitude')
ax2.set_ylabel('longitude')


fig.tight_layout()
plt.savefig('./output/a=13mape.svg')
plt.show()

# 定义数据
