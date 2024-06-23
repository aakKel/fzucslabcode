import numpy as np
from sklearn.linear_model import LassoLarsCV
from pack import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr

p = '../../dataset/36/pm2.5_bj_20140501-20150430_mean.csv'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
data = pd.read_csv(p)
# data = data.values.astype(np.float64)

a = [9,20,10,14]
# a = range(0,36)
fig = plt.figure(figsize=[10,6],dpi=300)

begin = 160
day = 3
res = np.zeros([len(a),day * 24])


for i in range(0,len(a)):
    data_tmp = data.iloc[(a[i] * 365) + begin :(a[i] * 365) + begin + day, :]
    data_tmp = data_tmp.values.astype(np.float64)
    y = []
    for j in range(day) :
        y.extend(data_tmp[j])
    res[i] = y




x = np.linspace(0,day*24 - 1,day * 24)
plt.xlabel('Time/hour')
plt.ylabel('PM2.5')
color = ['blue','red','green','black']
for i in range(0,len(a)):
    if i == 3 :
        plt.plot(x, res[i],c = 'black', alpha=1, lw=3, ls='-',label=('station' + str(a[i])))
    else :
        plt.plot(x,res[i],alpha=1, lw=2, ls=':',label = ('station' + str(a[i])))
plt.legend()

# plt.savefig('4-3sdiff.svg')
# plt.show()