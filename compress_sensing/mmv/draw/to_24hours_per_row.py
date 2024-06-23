import math

import pandas as pd
import numpy as np

p = '../dataset/shanghai2014_0_3station.csv'

data = pd.read_csv(p)

data = data.values.astype(np.float64)
(m,n) = data.shape
res_data = np.zeros([(int)(n/24) * 3,24],dtype=np.float64)
for i in range(0,m) :
    for j in range(0,n) :
        print(math.floor((i * n + j) / 24) , j % 24)
        res_data[math.floor((i * n + j) / 24)][j % 24] = data[i][j]
print(res_data)