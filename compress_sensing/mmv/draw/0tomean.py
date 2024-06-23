import pandas as pd
import numpy as np


p = '../dataset/shanghai2014_0.csv'

data = pd.read_csv(p)

data = data.values.astype(np.float64)
(m,n) = data.shape

for i in range(0,m) :
    for j in range(0,n):
        if data[i][j] == 0 :
            t = 0
            cnt = 0
            for k in range(max(0,j - 12),min(n - 1,j + 12)) :
                t += data[i][k]
                cnt += 1
            data[i][j] = t / cnt
pd.DataFrame(data).to_csv('../dataset/shanghai_24mean_3station.csv',index=False)