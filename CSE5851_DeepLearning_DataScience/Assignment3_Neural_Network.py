# Assignment 3
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

loss_hidden = [] # to record amount of loss for every hidden layer size

for n in range(2, 17):
    print(f'Now TRAINING with hidden size {n}...')
    K=2 # input layer size
    N=n # hidden layer size, can modify this parameter
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
                    WP[i,j] -= lr*EIP[j]*h[i]

            # hidden -> input weight updates
            for k in range(K):
                for i in range(N):
                    W[k,i] -= lr*EI[i]*x[iter,k]

        loss[epoch]=ER_SUM/len(x)

    loss_hidden.append(loss)

# 2. plot training loss according to different sizes of hidden layer(2 to 16)
for i in range(3):
    plt.plot(range(1, EPOCHS + 1), loss_hidden[i*5+0], 'b', label=f'hidden size {(i*5+0)+2}')
    plt.plot(range(1, EPOCHS + 1), loss_hidden[i*5+1], 'g', label=f'hidden size {(i*5+1)+2}')
    plt.plot(range(1, EPOCHS + 1), loss_hidden[i*5+2], 'r', label=f'hidden size {(i*5+2)+2}')
    plt.plot(range(1, EPOCHS + 1), loss_hidden[i*5+3], 'c', label=f'hidden size {(i*5+3)+2}')
    plt.plot(range(1, EPOCHS + 1), loss_hidden[i*5+4], 'm', label=f'hidden size {(i*5+4)+2}')

    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()