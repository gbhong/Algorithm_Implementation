# Assignment 2
# 2020312086 Hong Gibong

import numpy as np
import matplotlib.pyplot as plt

# initial settings
np.random.seed(0)
plt.style.use('seaborn')

# define functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

### data load & initialization ###
data = np.loadtxt('./data/training.txt')
np.random.shuffle(data)

x = data[:, :-1] # input
t = data[:, -1] # label
y = np.zeros((len(x),1)) # pred

K=2 # input layer size
N=8 # hidden layer size
M=1 # output layer size
lr=0.01 # learning rate is set to 0.01

batch_size=1 # batch size for SGD is set to 1
iter=int(len(x)/batch_size)

W = np.random.random((K,N)) # initialize weights between input and hidden
WP = np.random.random((N,M)) # initialize weights between hidden and output

### create output label ###
t_label = np.zeros((len(x),1))
for i in range(len(x)):
    if float(t[i]) == 1.:
        t_label[i] = 1

EPOCHS = 100
loss = np.zeros(EPOCHS) # to record amount of loss for every epoch.

for epoch in range(EPOCHS):
    ER_SUM = 0

    for iter in range(iter):
        ### forward ###
        # forward from input ---> hidden
        u = np.zeros(N)  # outputs after first weights matrix
        h = np.zeros(N)  # outputs from u1 after passing activation func.
        up = np.zeros(M)  # outputs after second weights matrix

        for i in range(N): # for every node in hidden layer
            for k in range(K): # for every node in input layer
                u[i] += W[k,i]*x[iter,k]
            h[i] = sigmoid(u[i])

        # forward from hidden -> output
        for j in range(M): # for every node in output layer
            for i in range(N): # for every node in hidden layer
                up[j] += WP[i,j]*h[i]
            y[iter,j] = sigmoid(up[j])

        ### calculate loss ###
        for i in range(M):  # for every output
            ER_SUM += (1. / 2.) * ((y[iter, i] - t_label[iter, i]) ** 2)

        ### backward(weight updates) ###
        # output -> hidden calculate gradient
        EIP = np.zeros(M)  # error*input between output and hidden
        for j in range(M):
            EIP[j]=(y[iter,j]-t_label[iter,j])*y[iter,j]*(1.-y[iter,j])

        # hidden -> input calculate gradient
        EI = np.zeros(N)  # error*input between hidden and input
        for i in range(N):
            for j in range(M):
                EI[i] += EIP[j] * WP[i,j] * h[i] * (1. - h[i])

        # output -> hidden weight updates
        for i in range(N):
            for j in range(M):
                WP[i,j] -= lr*(1/np.sqrt(EIP[j]**2+1e-8))*EIP[j]*h[i]

        # hidden -> input weight updates
        for k in range(K):
            for i in range(N):
                W[k,i] -= lr*(1/np.sqrt(EI[i]**2+1e-8))*EI[i]*x[iter,k]

    loss[epoch]=ER_SUM/len(x)

print(loss)
plt.plot(range(1, EPOCHS+1), loss, 'g', label='Loss per Epoch')
plt.show()