import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# 读取BedGraph文件
filename = "GSM1700639_hSCLC_ASCL1_H889_ChIP.bedGraph"
data = pd.read_csv(filename, sep="\t", header=None, names=["class", "x", "y", "score"])
data = data.sort_values("score", ascending=False)
data = data[1:100]
print(data)
# 计算每个区域的富集值

#
# 提取坐标和数值数据
x = data["x"]
y = data["y"]
score = data["score"]

# 创建一个二维数组来存储热力图数据
heatmap_data = pd.pivot_table(data, values="score", index="x", columns="y")
dpi = 300
fig= plt.subplots(figsize=(10, 10), dpi=dpi)

sns.set(font_scale=1.2)  # 设置字体大小
ax = sns.heatmap(heatmap_data, cmap='YlOrRd', cbar=False)
# ax.set_title("Heatmap")
ax.set_xlabel("")
ax.set_ylabel("")
# 绘制热力图
# sns.heatmap(heatmap_data, cmap='YlOrRd')
# plt.title("Heatmap")
# plt.xlabel("Y")
# plt.ylabel("X")
plt.show()
# data = data[0:100]

# data["enrichment"] = data["score"].cumsum()
# 绘制富集图
# plt.plot(data["end"], data["enrichment"])
# plt.show()