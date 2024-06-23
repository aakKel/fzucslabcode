import numpy as np

from pack import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_pearsonr(p, day, begin, station_num,flag,is_draw = False) :
    data = pd.read_csv(p)
    a = range(0,station_num)
    data_ori = np.zeros([len(a),day * 24])
    for i in range(0,len(a)):
        data_tmp = data.iloc[(a[i] * 365) + begin :(a[i] * 365) + begin + day, :]
        data_tmp = data_tmp.values.astype(np.float64)
        y = []
        for j in range(day) :
            y.extend(data_tmp[j])
        data_ori[i] = y
    res = np.zeros([len(a),len(a)])

    for i in range(0,len(a)):
        for j in range(0,len(a)):
            if flag == True:
                res[i][j] ,p_value = pearsonr(data_ori[i], data_ori[j])
            else :
                res[i][j], p_value = spearmanr(data_ori[i], data_ori[j])
    indices = np.zeros((len(a), len(a)), dtype=np.int64)
    for i in range(res.shape[0]):
        if is_draw == False:
            res[i][i] = -1
        row = res[i]
        indices[i] = np.argsort(-row)
    # print(res,indices)
    return res, indices

def getconn3max2medim1mn(choice):
    p = '../dataset/36/pm2.5_bj_20140501-20150430_mean.csv'
    station_num = 36
    begin = 100
    day = 7
    res, indices = get_pearsonr(p, day, begin, station_num, False,False)
    no_choice1 = []
    no_choice2 = []
    no_choice3 = []
    sum = []
    for i in range(0, len(indices)):
        sum.append(np.sum(res[i]))
    sum_idx = np.argsort(sum)

    for i in range(0, len(sum)):
        if i < len(sum) / 3:
            no_choice1.append(sum_idx[i])
        elif (i >= len(sum) / 3 and i < len(sum) * 2 / 3):
            no_choice2.append(sum_idx[i])
        elif (i >= len(sum) * 2 / 3):
            no_choice3.append(sum_idx[i])

    no_choice1_conn = np.zeros([len(no_choice1), 3], dtype=np.int64)
    no_choice2_conn = np.zeros([len(no_choice2), 3], dtype=np.int64)
    no_choice3_conn = np.zeros([len(no_choice3), 3], dtype=np.int64)
    print(no_choice1)

    for j in range(0, len(no_choice1)):
        cnt = 0
        for k in range(0, 36):
            if cnt < 3:
                if indices[no_choice1[j]][k] not in no_choice1:
                    no_choice1_conn[j][cnt] = indices[no_choice1[j]][k]
                    cnt += 1

    for j in range(0, len(no_choice2)):
        cnt = 0
        for k in range(0, 36):
            if cnt < 3:
                if indices[no_choice2[j]][k] not in no_choice2:
                    no_choice2_conn[j][cnt] = indices[no_choice2[j]][k]
                    cnt += 1

    for j in range(0, len(no_choice3)):
        cnt = 0
        for k in range(0, 36):
            if cnt < 3:
                if indices[no_choice3[j]][k] not in no_choice3:
                    no_choice3_conn[j][cnt] = indices[no_choice3[j]][k]
                    cnt += 1
    if choice == 1:
        return no_choice1
    if choice == 2:
        return no_choice2
    if choice == 3:
        return no_choice3


if __name__ == '__main__' :
    is_draw = False
    p = '../dataset/36/pm2.5_bj_20140501-20150430_mean.csv'
    station_num = 36
    begin = 100
    day = 7
    res, indices = get_pearsonr(p,day,begin,station_num,True,is_draw)
    no_choice1 = []
    no_choice2 = []
    no_choice3 = []

     # sum in row of res
    sum = []
    for i in range(0,len(indices)):
        sum.append(np.sum(res[i]))
    sum_idx = np.argsort(sum)
    high_idx = []
    low_idx = []
    fig = plt.subplots(figsize=(9, 6), dpi=300)
    for i in range(0,len(sum)):
        if i < len(sum) / 3 :
            plt.bar(sum_idx[i],sum[sum_idx[i]],color='lightblue')
            no_choice1.append(sum_idx[i])
            low_idx.append(sum_idx[i])
        elif (i >= len(sum) / 3 and i < len(sum) * 2 / 3):
            plt.bar(sum_idx[i], sum[sum_idx[i]], color='lightgreen')
            no_choice2.append(sum_idx[i])
        elif (i >= len(sum) * 2 / 3) :
            plt.bar(sum_idx[i], sum[sum_idx[i]], color='lavender')
            no_choice3.append(sum_idx[i])
            high_idx.append(sum_idx[i])

    no_choice1_conn = np.zeros([len(no_choice1), 3],dtype=np.int64)
    no_choice2_conn = np.zeros([len(no_choice2), 3], dtype=np.int64)
    no_choice3_conn = np.zeros([len(no_choice3), 3], dtype=np.int64)
    print(no_choice1)

    for j in range(0, len(no_choice1)):
        cnt = 0
        for k in range(0, 36):
            if cnt < 3:
                if indices[no_choice1[j]][k] not in no_choice1:
                    no_choice1_conn[j][cnt] = indices[no_choice1[j]][k]
                    cnt += 1

    for j in range(0, len(no_choice2)):
        cnt = 0
        for k in range(0, 36):
            if cnt < 3:
                if indices[no_choice2[j]][k] not in no_choice2:
                    no_choice2_conn[j][cnt] = indices[no_choice2[j]][k]
                    cnt += 1

    for j in range(0, len(no_choice3)):
        cnt = 0
        for k in range(0, 36):
            if cnt < 3:
                if indices[no_choice3[j]][k] not in no_choice3:
                    no_choice3_conn[j][cnt] = indices[no_choice3[j]][k]
                    cnt += 1

    filename = './output/npz/1_gen_station'+'.npz'
    np.savez(filename, A=no_choice1, B=no_choice1_conn,)
    filename = './output/npz/2_gen_station' + '.npz'
    np.savez(filename, A=no_choice2, B=no_choice2_conn, )
    filename = './output/npz/3_gen_station' + '.npz'
    np.savez(filename, A=no_choice3, B=no_choice3_conn, )



    # plt.bar(range(len(sum)), sum)
    plt.ylabel('Sum of Correlation')
    plt.xlabel('Station Number')
    plt.ylim(10, max(sum) + 1)
    # plt.savefig('./output/Sum of Correlation.svg')
    plt.show()
    # index_idx = np.argsort(sum)[::-1]
    # print(index_idx)
    # print(indices)
    # plt.figure(figsize=(20, 10), dpi=300)
    # plt.matshow(res, cmap=plt.get_cmap('Greens'), alpha=1)
    # plt.show()


    fig = plt.subplots(figsize = (9,9))
    heatmap = sns.heatmap(pd.DataFrame(res, index = range(1,37)),
                    annot=False, vmax=1,vmin = 0.3, xticklabels= False, yticklabels= False, square=True, cmap="YlGnBu",cbar=False)
    ax = heatmap.axes

    # 获取颜色条Axes对象
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(heatmap.collections[0], cax=cax)
    cbar.outline.set_visible(False)
    # 展示Heatmap
    # plt.savefig('./output/pearsonr.svg')
    plt.show()
    print(high_idx)
    print(low_idx)
