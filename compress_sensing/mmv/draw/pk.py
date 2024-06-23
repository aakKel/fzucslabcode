import pandas as pd
import numpy as np

p = '../../dataset/36/PM25_beijing_20140501-20150430.csv'

data = pd.read_csv(p)
data = data.iloc[:,1:]

data = data.values.astype(np.float64)

print(data)

res = np.zeros([int(data.shape[1] * data.shape[0] / 24),24])
cur = 0
for i in range(0,data.shape[1]) :
    tmp = data[:,i:i + 1]

    for j in range(0,365):
        t = 0
        for k in range(j * 24,(j + 1) * 24):
            res[cur][t] = tmp[k]
            t = t + 1
        cur = cur + 1

[m,n] = res.shape
for i in range(0,m) :
    for j in range(0,n):
        if res[i][j] == 0 :
            t = 0
            cnt = 0
            for k in range(0,24) :
                t += res[i][k]
                cnt += 1
            res[i][j] = t / cnt

print(res.shape)
print(res)

pd.DataFrame(res).to_csv('../../dataset/36/pm2.5_bj_20140501-20150430_mean.csv',index=False)