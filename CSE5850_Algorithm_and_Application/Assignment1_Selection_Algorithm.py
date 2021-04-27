# Assignment 1_Algorithm and App.
# 2020312086 Hong Gibong

# 3. Selection Algorithm

def combi(n:int, k:int):
    global i # define global variable to sum up
    if k == 0 or n == k:
        i += 1
    else:
        combi(n-1, k)
        combi(n-1, k-1)

i = 0
combi(4, 2)
print(i)