import numpy as np


filename = './output/npz/03-18-23-19/row_16, iter_1000, solve_num_10.npz'


data = np.load(filename)

A = data['A'].tolist()
B = data['B'].tolist()

print(len(A))
print((len(A[0][0])))
print(A[0][0])
