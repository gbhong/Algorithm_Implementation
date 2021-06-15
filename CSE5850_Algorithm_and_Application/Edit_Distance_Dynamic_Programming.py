# Final Project_Algorithm and App.
# 2020312086 Hong Gibong

# Find Levenshtein distance with various approaches
# Compare time complexity among approaches, and results following the user-specific cost for edit operation.

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

    def dist(self, mode='DP', cost=None, debug=False):
        '''
            choose approach between DP and Brute-force, for computing Levenshtein Distance
            returns optimal value and elapsed time.
        '''

        if self.str1 is None:
            self.str1 = self.randStr(N=self.M)
        if self.str2 is None:
            self.str2 = self.randStr(N=self.N)

        if mode == 'BF':
            t_start = time.time()
            dist = self.bf_(self.str1, self.str2)
            return (dist, time.time() - t_start)

        dp1_start = time.time()
        dp1_res = self.dp_(debug = debug)
        dp1_t = time.time() - dp1_start

        print('\n====================\n')

        dp2_start = time.time()
        dp2_res = self.dp_2(self.str1, self.str2, cost = cost, debug = debug)
        dp2_t = time.time() - dp2_start

        return dp1_res, dp2_res, dp1_t, dp2_t

    def dp_(self, debug = False):
        m, n = len(self.str1), len(self.str2)
        d = [[0] * (m + 1) for _ in range(n + 1)]

        # fill in matrix d through bottom-up approach

        for i in range(m + 1):
            for j in range(n + 1):
                if j == 0:
                    d[j][i] = i
                elif i == 0:
                    d[j][i] = j
                else:
                    if self.str1[i-1] == self.str2[j-1]:
                        subCost = 0
                    else:
                        subCost = 1

                    d[j][i] = min(d[j-1][i]+1,
                                  d[j][i-1]+1,
                                  d[j-1][i-1]+subCost)

        if debug:
            for j in range(1, n+1):
                print(d[j][1:])

        return d[n][m]

    def dp_2(self, str1, str2, cost = None, debug = False):
        '''
            Computing Edit distance with only two matrix rows to reduce space complexity.
            Caching results of only previous and current rows.
        '''
        m, n = len(str1), len(str2)

        if m < n:
            return self.dp_2(str2, str1, cost=cost, debug=debug)

        if n == 0:
            return m

        if cost is None:
            cost = {}

        def substitution_cost(c1, c2):
            if c1 == c2:
                return 0
            return cost.get((c1, c2), 1) # if there is no key in dict, return 1

        previous_row = range(n + 1)
        for i, c1 in enumerate(str1):
            current_row = [i + 1]

            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + substitution_cost(c1, c2)
                current_row.append(min(insertions, deletions, substitutions))

            if debug:
                print(current_row[1:])

            previous_row = current_row

        return previous_row[-1]

    def bf_(self, str1, str2):
        if len(str1) == 0:
            return len(str2)
        elif len(str2) == 0:
            return len(str1)

        if str1[0] == str2[0]:
            return self.bf_(str1[1:], str2[1:])

        return 1 + min(self.bf_(str1, str2[1:]),
                       self.bf_(str1[1:], str2),
                       self.bf_(str1[1:], str2[1:])
                       )


if __name__ == '__main__':
    # input_list = [100, 300, 500, 700, 900]
    input_list = None

    if input_list is None: # to examine the effect of weighted edit costs
        editdist = LevDistance(str1='kitten', str2='sitting')

        # without user-defined edit costs
        dist1, dist2, t1, t2 = editdist.dist(mode='DP', debug=True)
        print('\t'.join(['edit dist for DP with full matrix: ' + str(dist1),
                         'edit dist for DP with 2 rows matrix: ' + str(dist2)]))

        print('\n===========================\n')

        # after define user-specific edit costs
        cost = {('i', 'e'): 0.1,
                ('e', 'i'): 0.1}

        dist1, dist2, t1, t2 = editdist.dist(mode='DP', cost=cost, debug=True)
        print('\t'.join(['edit dist for DP with full matrix: ' + str(dist1),
                         'edit dist for DP with 2 rows matrix: ' + str(dist2)]))

    else: # to examine time complexity
        time_rec_dp, time_rec_bf = [], []
        for i in input_list:
            editdist = LevDistance(M=i, N=i) # assign valid strings, or integers for randomly generated strings

            dist1, dist2, t1, t2 = editdist.dist(mode='DP')
            print('\t'.join([str(i), str(dist1), str(dist2)]))
            time_rec_dp.append(t1)

            # dist, elapsed_time = editdist.dist(mode='BF')
            # print('\t'.join([str(i), str(dist), str(elapsed_time)]))
            # time_rec_bf.append(elapsed_time)

        n = [time_rec_dp[0] * (i / 100) ** 2 for i in input_list]

        # np = [0 for _ in range(len(input_list))]
        # for idx in range(len(input_list)):
        #     if idx == 0:
        #         np[idx] = time_rec_bf[0]
        #     else:
        #         np[idx] = time_rec_bf[idx-1] * 3

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(input_list)), time_rec_dp, 'red', label='Elapsed Time for DP')
        plt.plot(range(len(input_list)), n, 'green', label='O(n^2)')

        # plt.plot(range(len(input_list)), time_rec_bf, 'blue', label='Elapsed Time for BF')
        # plt.plot(range(len(input_list)), np, 'orange', label='O(3^n)')

        plt.title('Comparison on Elapsed Time following input size')
        plt.xlabel('Input Size')
        plt.ylabel('Elapsed Time')
        plt.xticks(ticks=range(len(input_list)), labels=input_list)
        plt.legend()
        plt.show()