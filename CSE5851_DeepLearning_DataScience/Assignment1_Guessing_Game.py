# Assignment 1
# 2020312086 Hong Gibong

# 1. Guessing Game
import random
num = random.randint(1, 9) # generate a random integer

cnt = 0
while True: # keep the game going until the user types 'exit'
    a = input('guess random number: ')
    if a == 'exit':
        print(f'you have taken {cnt} guesses')
        break
    else:
        cnt += 1 # keep track of the number of guesses user has taken
        a = int(a) # make a as integer to compare with input number
        if a == num:
            print('exactly right')
        elif a >= num:
            print('too high')
        else:
            print('too low')