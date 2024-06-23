from pearsonr import get_pearsonr
import random
import numpy as np
import datetime
def get_random_npz(p, day, begin_day, num):
    # p = '../dataset/36/pm2.5_bj_20140501-20150430_mean.csv'
    res, indices = get_pearsonr(p, day, begin_day, num,False)
    print(res)
    print(indices)

    randnum = random.randint(0, 1 << num - 1)
    print(bin(randnum))
    print(randnum)
    A = []
    for i in range(0,num):
        if ((randnum >> i) & 1) == 1 :
            A.append(i)
    print(A)

    B = []
    for i in range(len(A)):
        cnt = 0
        tmp = []
        for k in range(len(indices[i])):
            if indices[i][k] not in A :
                tmp.append(indices[i][k])
                cnt += 1
            if cnt == 3:
                break
        B.append(tmp)
    print(B)
    filename = './output/npz/' + datetime.datetime.now().strftime('output_%Y%m%d%H%M%S') + '.npz'
    np.savez(filename, A=A, B=B)
    return filename

def get_greedy_npz(p, day, begin_day, num):
    high_idx = [16, 19, 12, 22, 2, 15, 6, 8, 10, 5, 9, 20]
    low_idx = [7, 34, 14, 28, 25, 30, 32, 33, 29, 23, 27, 0]
    no_sample = []
    for i in range(0,36):
        if i not in high_idx :
            no_sample.append(i)
    # r1 = random.sample(high_idx, (int)(len(high_idx) / 3 * 2))
    # r2 = random.sample(low_idx, (int)(len(low_idx) / 3 * 2))
    r = random.sample(no_sample, (int)(len(no_sample) / 3 * 2))
    A = sorted(r)
    res, indices = get_pearsonr(p, day, begin_day, num, False)
    B = []
    for i in range(len(A)):
        cnt = 0
        tmp = []
        for k in range(len(indices[i])):
            if indices[i][k] not in A:
                tmp.append(indices[i][k])
                cnt += 1
            if cnt == 3:
                break
        B.append(tmp)
    print(B)
    filename = './output/npz/' + datetime.datetime.now().strftime('output_%Y%m%d%H%M%S') + '.npz'
    np.savez(filename, A=A, B=B)
    return filename

if __name__ == '__main__' :
    p = '../dataset/36/pm2.5_bj_20140501-20150430_mean.csv'
    day = 7
    begin = 100
    num = 36
    get_greedy_npz(p, day, begin, num)