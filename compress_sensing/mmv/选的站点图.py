import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

choice = [[9, 11, 20],
          [7, 8, 9, 11, 12],
          [6, 8, 9, 11, 14, 19, 21],
          [2, 5, 6, 8, 9, 11, 14, 19, 21],
          [1, 3, 6, 7, 9, 10, 11, 14, 15, 19, 21],
          [1, 2, 3, 6, 7, 8, 10, 12, 13, 14, 17, 19, 21],
          ]



(fig,axs) = plt.subplots(2, 3, figsize=(32, 20),dpi = 300)


p = '../dataset/36/pm25_latlng.csv'
data = pd.read_csv(p)
data = data.iloc[:, 1:]
data = data.values.astype(np.float64)

# pp = [3,5,7,9,11,13]
for i, ax in enumerate(axs.flatten()):
    f = False
    for j in range(36):
        if j in choice[i]:
            if f == False:
                ax.scatter(data[j][0], data[j][1], color='red', label='selected')
                f = True
            else :
                ax.scatter(data[j][0], data[j][1], color='red')
        else :
            ax.scatter(data[j][0], data[j][1], color='black')
    ax.set_title('a = '+str((i + 1)*2+1))
    ax.legend()
    ax.set_xlabel('latitude')
    ax.set_ylabel('longitude')
fig.tight_layout()
# plt.savefig('./output/choice.svg')
plt.show()
