# Assignment 1_Algorithm and App.
# 2020312086 Hong Gibong

# 2. Restricted Tower of Hanoi

class hanoi(object):
    def __init__(self):
        self.answer = [] # Create empty list to follow every move
        self.moves = 0 # Count number of moves

    def towers(self, n:int, source:int, dest:int, aux:int):
        if n == 1:
            self.answer.extend([[source, aux], [aux, dest]]) # use 'extend' method as inputs are multiple lists
            self.moves += 2 # add 2 because every plate is moved via 'Peg B'
        else:
            self.towers(n-1, source=source, dest=dest, aux=aux) # move n-1 plates from Peg A to Peg C via Peg B

            self.answer.append([source, aux]) # move the biggest plate from Peg A to Peg B
            self.moves += 1

            self.towers(n-1, source=dest, dest=source, aux=aux) # move n-1 plates from Peg C to Peg A via Peg B

            self.answer.append([aux, dest]) # move the biggest plate from Peg B to Peg C
            self.moves += 1

            self.towers(n-1, source=source, dest=dest, aux=aux) # move n-1 plates from Peg A to Peg C via Peg B

    def solution(self, n:int):
        # n disks should b moved from peg A to peg C and mulst involve peg B for every move.
        self.towers(n, source='A', dest='C', aux='B')
        return self.answer, self.moves

hanoi = hanoi()
seq_moves, num_moves = hanoi.solution(3)
print(f'The sequence of moves is {seq_moves} while resulting {num_moves} moves.')