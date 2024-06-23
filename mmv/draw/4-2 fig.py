import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
p = '../../dataset/36/pm25_latlng.csv'

data = pd.read_csv(p)
data_tmp_v = data.values.astype(np.float64)
a = [9,20,10,14]


filename = '../output/npz/iter1000,n6output_20240104095434.npz'
# filename = '../output/npz/iter1000,n6output_20240104082110.npz'
# filename = '../output/npz/iter1000,n6output_20240118015714.npz'
data = np.load(filename)
#
# # 从data中读取矩阵A和B
A = data['A'].tolist()
B = data['B'].tolist()
C = data['C'].tolist()
# D = data['D']
a = A
data_tmp_v = data_tmp_v[:,1:]
def plot_points(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    colors = np.random.rand(len(points), 3)
    fig, ax = plt.subplots(figsize=(6, 6),dpi=300)
    ax.scatter(x, y,s=13,c='black')
    plt.yticks(fontproperties='Times New Roman')
    plt.xticks(fontproperties='Times New Roman')
    # ax.scatter(x[9], y[9], s=13, c='blue')
    # ax.scatter(x[20], y[20], s=13, c='orange')
    # ax.scatter(x[10], y[10], s=13, c='green')
    # ax.scatter(x[14], y[14], s=13, c='black')
    for i, point in enumerate(points):
        if i in a:
            ax.scatter(point[0],point[1], s=13)
            ax.annotate(str(i), (point[0], point[1]), textcoords="offset points", xytext=(0,5), ha='center',fontsize=7)
    plt.ylabel('Longitude',fontproperties={'family':'Times New Roman'},fontsize=12)
    plt.xlabel('Latitude',fontproperties={'family':'Times New Roman'},fontsize=12)

    for i in range(len(A)):
        random_color = f'#{np.random.randint(0, 0xFFFFFF):06x}'
        ax.scatter(points[A[i]][0], points[A[i]][1], c=random_color)
        for j in range(len(B[i])):
            x_t = [points[A[i]][0], points[B[i][j]][0]]
            y_t = [points[A[i]][1], points[B[i][j]][1]]
            ax.plot(x_t, y_t, c=random_color)


    plt.tight_layout()
    plt.show()
    # plt.savefig('4-7-new.svg')
plot_points(data_tmp_v)
# print(data_tmp_v)