# # Assignment 4_Algorithm and App.
# # 2020312086 Hong Gibong

# Q1 Strassen's Algorithm

# size n의 정방행렬 생성
def initMatrix(n):
    return [[0 for _ in range(n)] for _ in range(n)]

def add(M1, M2, n):
    temp = initMatrix(n)
    for i in range(n):
        for j in range(n):
            temp[i][j] = M1[i][j] + M2[i][j]
    return temp

def subtract(M1, M2, n):
    temp = initMatrix(n)
    for i in range(n):
        for j in range(n):
            temp[i][j] = M1[i][j] - M2[i][j]
    return temp

def strassen(A, B, n):
    if n == 1:
        C = initMatrix(1)
        C[0][0] = A[0][0] * B[0][0]
        return C

    C = initMatrix(n)
    k = n // 2

    A11 = initMatrix(k)
    A12 = initMatrix(k)
    A21 = initMatrix(k)
    A22 = initMatrix(k)
    B11 = initMatrix(k)
    B12 = initMatrix(k)
    B21 = initMatrix(k)
    B22 = initMatrix(k)

    for i in range(k):
        for j in range(k):
            A11[i][j] = A[i][j]
            A12[i][j] = A[i][k+j]
            A21[i][j] = A[k+i][j]
            A22[i][j] = A[k+i][k+j]

            B11[i][j] = B[i][j]
            B12[i][j] = B[i][k + j]
            B21[i][j] = B[k + i][j]
            B22[i][j] = B[k + i][k + j]

    P1 = strassen(A11, subtract(B12, B22, k), k)
    P2 = strassen(add(A11, A12, k), B22, k)
    P3 = strassen(add(A21, A22, k), B11, k)
    P4 = strassen(A22, subtract(B21, B11, k), k)
    P5 = strassen(add(A11, A22, k), add(B11, B22, k), k)
    P6 = strassen(subtract(A12, A22, k), add(B21, B22, k), k)
    P7 = strassen(subtract(A11, A21, k), add(B11, B12, k), k)

    C11 = subtract(add(add(P5, P4, k), P6, k), P2, k)
    C12 = add(P1, P2, k)
    C21 = add(P3, P4, k)
    C22 = subtract(subtract(add(P5, P1, k), P3, k), P7, k)

    for i in range(k):
        for j in range(k):
            C[i][j] = C11[i][j]
            C[i][j+k] = C12[i][j]
            C[k+i][j] = C21[i][j]
            C[k+i][k+j] = C22[i][j]

    return C

A = [[1,0,2,1],[4,1,1,0],[0,1,3,0],[5,0,2,1]]
B = [[0,1,0,1],[2,1,1,4],[2,0,1,1],[1,3,5,0]]
print(strassen(A, B, n=len(A)))