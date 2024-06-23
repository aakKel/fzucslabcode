import numpy as np
from pack import *
L = 3
m,n = 2,5

Psi = trip1(n)
x = np.array([[1,2,3,4,5],[6,7,8,9,10],[1,3,4,5,7]])
print(x)
f1 = np.array([[0,0,0,1,0],[0,0,1,0,0]])
f2 = np.array([[0,1,0,0,0],[0,0,0,0,1]])
f3 = np.array([[1,0,0,0,0],[0,1,0,0,0]])

orif = [f1,f2,f3]

Y = []
for i in range(0,L) :
    Y.extend(np.dot(orif[i],x[i].T))

f1 = np.dot(f1,Psi)
f2 = np.dot(f2,Psi)
f3 = np.dot(f3,Psi)
matrices = [f1, f2, f3]
fi = np.block([[matrices[i] if i == j else np.zeros((m, n)) for j in range(len(matrices))] for i in range(len(matrices))])

xk = np.dot(np.linalg.pinv(fi),Y)


tiny_error = 0.2   # 惩罚项      参数
p = 0.5            # 0<= p <=1  参数
min_error = 0.5     # 终止的误差限 参数
LD = 0.01           # 考虑噪声误差 参数
while 1:
    diag_matrix = np.zeros([n, n])
    for i in range(0, n):
        diag_matrix[i][i] = np.power(np.sqrt(np.linalg.norm(np.sum(xk[i])) + tiny_error), 1 - p / 2)
    Wk = np.kron(np.eye(L),diag_matrix)
    Ak = np.dot(fi,Wk)
    Ak_tmp = np.dot(Ak,np.conj(Ak).T)
    new_I = np.multiply(LD,np.eye(Ak_tmp.shape[0]))
    new_xk = np.dot(np.dot(Wk,np.conj(Ak).T),np.dot((np.linalg.inv(np.add(Ak_tmp,new_I))),Y))

    if (np.linalg.norm(np.subtract(new_xk,xk)) / np.linalg.norm(xk)) < min_error :
        xk = new_xk
        break
    xk = new_xk

res_x = np.reshape(xk,(L,n))

for i in range(0,L) :
    res_x[i] = np.dot(Psi,res_x[i])
print(res_x)


for i in range(0,L) :
    for j in range(0,m) :
        for k in range(0,n):
            if orif[i][j][k] == 1:
                res_x[i][k] = x[i][k]
print(res_x)
mape = 0
cnt = 0
for i in range(0,L):
    for j in range(0,n) :
        cnt += 1
        mape += np.abs(x[i][j] - res_x[i][j]) / x[i][j]
mape = (mape / cnt) * 100
print(mape)



t = []
for i in range(3):
    t.append(np.random.randn(2,3))
print(t)
for i in range(3):
    print(t[i])
