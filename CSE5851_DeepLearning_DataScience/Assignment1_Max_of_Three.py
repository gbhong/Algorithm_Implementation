# Assignment 1
# 2020312086 Hong Gibong

# 4. Max of Three
def big_three(a:int, b:int, c:int):
    if a >= b:
        if a >= c:
            return a
        else:
            return c
    else:
        if b >= c:
            return b
        else:
            return c

print(big_three(3,5,2))
print(big_three(4,3,3))