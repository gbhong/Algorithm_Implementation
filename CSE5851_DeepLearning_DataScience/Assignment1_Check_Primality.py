# Assignment 1
# 2020312086 Hong Gibong

# 2. Check Primality
import random
num = random.randint(1, 1000) # generate a random integer
print(f'randomly chosen number is {num}')

if num == 1: # number 1 is prime
    print(f'number {num} is prime')
else:
    d = num - 1 # check divisor below chosen number
    while d > 1:
        if num % d == 0: # if num is divided by d, num is not prime
            print(f'number {num} is not prime')
            break
        else:
            d -= 1 # check with next divisor

    if d == 1: # if num in divided only by 1 except for itself, then it is prime
        print(f'number {num} is prime')
