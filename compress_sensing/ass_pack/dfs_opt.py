import numpy as np


class CSSP:
    def __init__(self ,sensing_matrix ,need_col_num):
        self.sensing_matrix = sensing_matrix
        self.need_col_num = need_col_num
        self.min_error = 1e3
        self.min_error_list = []
        self.max_thersh = float(1e-12)
        self.max_iterator_time = 1e4
        self.current_iterator_time = 0
    # 计算当前选择的列的误差
    def get_error(self ,choice_list):
        get_col = []
        for i in range(0,len(choice_list)):
            if choice_list[i] == 1:
                get_col.append(i)

        choice_col_matrix = self.sensing_matrix[:, get_col]
        this_iter_error = self.sensing_matrix - np.dot(np.dot(choice_col_matrix, np.linalg.pinv(choice_col_matrix)),self.sensing_matrix)
        this_iter_error = np.linalg.norm(this_iter_error, ord=None, axis=None, keepdims=False)
        return this_iter_error

    #dfs 递归计算，如果当前误差比最小误差大3个数量级 剪支
    def dfs(self,choice_list):
        if np.sum(choice_list) == self.need_col_num:
            current_error = self.get_error(choice_list)

            if self.min_error > current_error:
                self.min_error = current_error
                self.min_error_list = choice_list.copy()

                print("min_error_list",self.min_error_list)
                print("min_error",self.min_error)
                print("current_iterator_time",self.current_iterator_time)
            return True

        for i in range(0,len(choice_list)):
            if choice_list[i] != 1:
                if self.current_iterator_time > self.max_iterator_time:
                    break
                self.current_iterator_time += 1
                choice_list[i] = 1
                self.dfs(choice_list)
                choice_list[i] = 0

    def run_dfs(self):
        n = self.sensing_matrix.shape[1]

        choice_list = np.zeros(n,dtype=np.int32)

        for i in range(0,self.need_col_num):
            choice_list[i] = 1
        current_error = 1

        while current_error > self.max_thersh:
            if self.current_iterator_time > self.max_iterator_time:
                break
            self.current_iterator_time += 1
            np.random.shuffle(choice_list)
            #self.print_choiced_col(choice_list)

            current_error = self.get_error(choice_list)
            if self.min_error > current_error:
                self.min_error = current_error
                self.min_error_list = choice_list.copy()
            #print("current_error",current_error)

        #self.min_error_list = choice_list.copy()
        print("min_error_list", self.min_error_list)
        #self.min_error = current_error

        #self.dfs(choice_list)
        return self.print()

    def print(self):
        for i in range(0, len(self.min_error_list)):
            if self.min_error_list[i] == 0:
                self.sensing_matrix[:, i] = 0

        # print("need_col_num", self.need_col_num)
        # print("min_error_list", self.min_error_list)
        # print("min_error",self.min_error)
        return self.sensing_matrix

    def print_choiced_col(self,choice_list):
        res = []
        for i in range(0,len(choice_list)):
            if choice_list[i] == 1:
                res.append(i)
        print("need col :",self.need_col_num,"currently selected col",res)

if __name__ == '__main__':
    a = np.array([[1,2,3,5,3,2],[2,2,3,3,4,5],[3,2,4,4,5,2]])

    A = CSSP(a,2).run_dfs()

    print(A)
