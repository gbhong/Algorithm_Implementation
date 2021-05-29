# Final Project_Algorithm and App.
# 2020312086 Hong Gibong

# Levenshtein distance

class LevDistance(object):
    '''
        computes string similarity, based on the minimum edit counts of change from one string to the other.
    '''
    def __init__(self, str1, str2):
        self.str1 = str1
        self.str2 = str2

    def dist(self, mode):
        '''
            choose approach between DP and Brute-force, for computing Levenshtein Distance
        '''
        if mode == 'DP':

    def dp_(self):
        d = [[0]*(len(self.str2)) for _ in range(len(self.str1))]

        for i in range(len(self.str2)):
            