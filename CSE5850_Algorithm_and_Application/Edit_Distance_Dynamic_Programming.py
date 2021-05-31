# Final Project_Algorithm and App.
# 2020312086 Hong Gibong

# Levenshtein distance
# todo: build code to compare time complexity among approaches

import random
import string
import time
import matplotlib.pyplot as plt

class LevDistance(object):
    '''
        computes string similarity, based on the minimum edit counts of change from one string to the other.
    '''
    def __init__(self, str1=None, str2=None, M=None, N=None):
        self.str1 = str1
        self.str2 = str2
        self.M = M
        self.N = N

    def randStr(self, chars = string.ascii_lowercase, N = 10):
        return ''.join(random.choice(chars) for _ in range(N))

    def dist(self, mode='DP'):
        '''
            choose approach between DP and Brute-force, for computing Levenshtein Distance
        '''
        if self.str1 is None:
            self.str1 = self.randStr(N=self.M)
        if self.str2 is None:
            self.str2 = self.randStr(N=self.N)

        if mode == 'BF':
            return self.bf_(self.str1, self.str2)

        return self.dp_()

    def dp_(self):
        m, n = len(self.str1), len(self.str2)
        d = [[0]*(n+1) for _ in range(m+1)]

        # fill in matrix d through bottom-up approach
        for j in range(n+1):
            for i in range(m+1):
                if i == 0:
                    d[i][j] = j
                elif j == 0:
                    d[i][j] = i
                else:
                    if self.str1[i-1] == self.str2[j-1]:
                        subCost = 0
                    else:
                        subCost = 1

                    d[i][j] = min(d[i-1][j]+1,
                                  d[i][j-1]+1,
                                  d[i-1][j-1]+subCost)

        return d[m][n]

    def bf_(self, str1, str2):
        if len(str1) == 0:
            return len(str2)
        elif len(str2) == 0:
            return len(str1)

        if str1[-1] == str2[-1]:
            return self.bf_(str1[:-1], str2[:-1])

        return 1 + min(self.bf_(str1, str2[:-1]),
                       self.bf_(str1[:-1], str2),
                       self.bf_(str1[:-1], str2[:-1])
                       )

if __name__ == '__main__':
    input_list = [100, 300, 500, 700, 900]

    time_rec_dp, time_rec_bf = [], []
    for i in input_list:
        editdist = LevDistance(M=i, N=i) # assign valid strings, or integers for randomly generated strings

        t_start = time.time()
        print(editdist.dist(mode='DP'))
        elapsed_time = time.time() - t_start
        print(i, '\t', elapsed_time)
        time_rec_dp.append(elapsed_time)

        t_start = time.time()
        print(editdist.dist(mode='BF'))
        elapsed_time = time.time() - t_start
        print(i, '\t', elapsed_time)
        time_rec_bf.append(elapsed_time)

    n = [time_rec_dp[0] * (i / 100) ** 2 for i in input_list]
    np = [3 ** (time_rec_bf[0] * (i / 100)) for i in input_list]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(input_list)), time_rec_dp, 'r', label='Elapsed Time for DP')
    plt.plot(range(len(input_list)), time_rec_bf, 'b', label='Elapsed Time for BF')

    plt.plot(range(len(input_list)), n, 'green', label='O(n^2)')
    plt.plot(range(len(input_list)), np, 'orange', label='O(3^n)')

    plt.title('Comparison on Elapsed Time following input size')
    plt.xlabel('Input Size')
    plt.ylabel('Elapsed Time')
    plt.xticks(ticks=range(len(input_list)), labels=input_list)
    plt.legend()
    # plt.show()
    plt.savefig('./edit_dist_plot.png', dpi=300)