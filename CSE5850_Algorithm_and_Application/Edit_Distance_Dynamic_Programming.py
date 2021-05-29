# Final Project_Algorithm and App.
# 2020312086 Hong Gibong

# Levenshtein distance
# todo: build code to compare time complexity among approaches

import random
import string
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

    def plot_(self):

if __name__ == '__main__':
    EditDist = LevDistance(M=10, N=10) # assign valid strings, or integers for randomly generated strings
    print(EditDist.dist())