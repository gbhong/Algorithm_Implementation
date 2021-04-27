# Assignment 3_Algorithm and App.
# 2020312086 Hong Gibong

import time
import numpy as np
import matplotlib.pyplot as plt

# 2. Sorting - bubblesort / quicksort

class sorting(object):
    def __init__(self, N:int, mode:str):
        self.N = N
        self.arr = np.random.randint(1, 10000, size=self.N) # initialization
        self.mode = mode

    def bubblesort(self):
        for n in range(self.N-1, 0, -1):
            for i in range(n):
                if self.arr[i] > self.arr[i+1]:
                    self.arr[i], self.arr[i+1] = self.arr[i+1], self.arr[i]

    def quicksort(self, arr:list, low:int, high:int):
        if low < high:
            pivot = self.partition(arr, low, high)
            self.quicksort(arr, low, pivot - 1)
            self.quicksort(arr, pivot + 1, high)

    def partition(self, arr:list, pivot:int, high:int):
        i = pivot + 1; j = high
        while True:
            while i < high and arr[i] < arr[pivot]:
                i += 1
            while j > pivot and arr[j] > arr[pivot]:
                j -= 1
            if j <= i:
                break
            arr[i], arr[j] = arr[j], arr[i]
            i += 1; j -= 1

        arr[pivot], arr[j] = arr[j], arr[pivot]
        return j

    # def quicksort(self, ARRAY):
    #     ARRAY_LENGTH = len(ARRAY)
    #     if (ARRAY_LENGTH <= 1):
    #         return ARRAY
    #     else:
    #         PIVOT = ARRAY[0]
    #         GREATER = [element for element in ARRAY[1:] if element > PIVOT]
    #         LESSER = [element for element in ARRAY[1:] if element <= PIVOT]
    #         return self.quicksort(LESSER) + [PIVOT] + self.quicksort(GREATER)

    def main(self):
        if self.mode == 'bubble':
            self.bubblesort()
        elif self.mode == 'quick':
            self.quicksort(arr=self.arr, low=0, high=self.N-1)
            # self.quicksort(self.arr)

if __name__=="__main__":
    for mode in ['bubble', 'quick']:
        result = [] # keep track of results for each type of sorting algorithm
        for k in range(100, 2100, 200):
            print(f'Now on {mode} sort mode, input size {k}')
            start = time.time()
            sorting(N=k, mode=mode).main()
            result.append(time.time()-start)

        if mode == 'bubble':
            n2 = [result[0] * (i/100) ** 2 for i in range(100, 2100, 200)]

            plt.figure(figsize=(12, 6))
            plt.plot(range(10), n2, 'r', label='O(n^2)')

        elif mode == 'quick':
            nln = [result[0] * (i/100) * np.log(i/101) for i in range(100, 2100, 200)]
            nlnp = [result[0] * (i / 100) * np.log(i / 100) * 0.1 for i in range(100, 2100, 200)]

            plt.figure(figsize=(12, 6))
            plt.plot(range(10), nln, 'g', label='O(n^logn)')
            plt.plot(range(10), nlnp, 'r', label='O(0.1n^logn)')

        plt.plot(range(10), result, 'b', label='Elapsed Time')
        plt.title('Comparison on Elapsed Time following input size')
        plt.xlabel('Input Size')
        plt.ylabel('Elapsed Time')
        plt.xticks(ticks=range(len(range(100, 2100, 200))), labels=range(100, 2100, 200))
        plt.legend()
        plt.show()