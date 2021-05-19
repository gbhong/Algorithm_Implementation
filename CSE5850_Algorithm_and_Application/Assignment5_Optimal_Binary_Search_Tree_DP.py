# Assignment 5_Algorithm and App.
# 2020312086 Hong Gibong

# Q2 Binary Search Tree

import numpy as np

def optimalBST(p,q,n):

    e = np.zeros((n+1,n+1))
    w = np.zeros((n+1,n+1))
    root = np.zeros((n+1,n+1))

    # Initialization
    for i in range(n+1):
        e[i,i] = q[i]
        w[i,i] = q[i]
    for i in range(0,n):
        root[i,i] = i+1

    for l in range(1,n+1):
        for i in range(0, n-l+1):
            j = i+l
            min_ = np.math.inf
            w[i,j] = w[i,j-1] + p[j] + q[j]
            for r in range(i,j):
                t = e[i, r-1+1] + e[r+1,j] +  w[i,j]
                if t < min_:
                    min_ = t
                    e[i, j] = t
                    root[i, j-1] = r+1

    root_pruned = np.delete(np.delete(root, n, 1), n, 0)        # Trim last col & row.

    print("------ main table -------")
    print(e)
    print("------ w -------")
    print(w)
    print("----- root table -----")
    print(root_pruned)

def main():
    # p = [0,.15,.1,.05,.1,.2]
    # q = [.05,.1,.05,.05,.05,.1]
    p = [0, 0.04, 0.06, 0.08, 0.02, 0.10, 0.12, 0.14]
    q = [0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05]
    n = len(p)-1

    optimalBST(p,q,n)

if __name__ == '__main__':
    main()
