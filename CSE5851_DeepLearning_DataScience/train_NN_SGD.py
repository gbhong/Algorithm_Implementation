# Code Implementation for Simple Neural Network

import numpy as np
import matplotlib.pyplot as plt

# initial settings
np.random.seed(0)
plt.style.use('seaborn')

class NN_SGD(object):
    '''
    Args:
        input_size (:obj:'int')
    '''
    def __init__(self, hidden_size, lr, EPOCHS, optimizer='SGD'):
        self.hidden_size = hidden_size
        self.lr = lr
        self.EPOCHS = EPOCHS
        self.optimizer = optimizer

    def initialize(self, X, labels):
        '''
        Creating a network(weights) based on user-defined inputs.
        :return: initialized weights for neural network
        '''
        return np.random.random((self.input_size, self.hidden_size)), np.random.random((self.hidden_size, self.output_size))

    def forward(self, weights, inputs, weights_size, input_size):
        '''
        Implementing the forward propagation for a row of data.
        All of the outputs from one layer become inputs to the neurons on the next layer.

        Args:
        - weights: matrix, with dimension 'hidden size * batch size'
        - inputs: inputs with given batch size
        :return: inputs for next forward network
        '''
        u1 = np.zeros(weights_size) # array containing outputs after matrix multiplication
        u2 = np.zeros(weights_size) # array containing outputs after passing activation func.

        for j in range(weights_size):
            for i in range(input_size):
                u1[j] = weights[i,j]*inputs[i]
            u2[j] = 1/(1+np.exp(-u1[j])) # sigmoid for activation func.

        return u2 # inputs for next feed-forward network

    def backward_loss(self, preds, labels, labels_size):
        '''
        Args:
            - preds: predicted float values
            - labels: true values
        :return: matrix containing losses for each label.
        '''
        EI = np.zeros(labels_size)
        for j in range(labels_size):
            EI[j]=(preds[j]-labels[j])*preds[j]*(1.-preds[j])
        return EI

    def backward(self, hidden, weights, errors, hidden_size, errors_size):
        '''
        Calculate gradient before updating weights
        '''
        EI = np.zeros(hidden_size)
        for i in range(hidden_size):
            for j in range(errors_size):
                EI[i] += errors[j] * weights[i,j] * hidden[i] * (1. - hidden[i])
        return EI

    def update_weights(self, weights, errors, inputs):
        for i in range(len(inputs)):
            for j in range(len(errors)):
                weights[i,j] -= self.lr * errors[j] * inputs[i]

    def get_loss(self, preds, labels, labels_size, mode):
        if mode == 'train':
            for j in range(labels_size):
                self.losses += (1. / 2.) * ((preds[j] - labels[j]) ** 2)
        elif mode == 'eval':
            for j in range(labels_size):
                self.val_losses += (1. / 2.) * ((preds[j] - labels[j]) ** 2)

    def fit(self, X, labels, eval_set):
        '''
        ARGS:
            - eval_set: (obj:tuple): contains test set, which is already separated into features and label.
        '''
        X_val, y_val = eval_set[0], eval_set[1]

        # transfer labels into one-hot vector form
        train_labels, test_labels = np.zeros((len(X), 1)), np.zeros((len(X_val), 1))
        train_labels = self.one_hot_vectorize(labels, train_labels)
        test_labels = self.one_hot_vectorize(y_val, test_labels)

        self.input_size, self.output_size = len(X[0]), len(train_labels[0])
        print(f'input size:{self.input_size}, hidden size:{self.hidden_size}, output size:{self.output_size}')
        print()

        # initialize weights for feed-forward network
        self.W1, self.W2 = self.initialize(X, labels)

        # create empty array to contain predicted values for every row
        self.train_pred, self.val_pred = np.zeros((len(X), 1)), np.zeros((len(X_val), 1))

        # create empty array to record losses for every epoch
        self.train_loss, self.val_loss = np.zeros(self.EPOCHS), np.zeros(self.EPOCHS)

        for epoch in range(1, self.EPOCHS+1):
            self.losses = 0. # initialize loss value for every epoch
            self.val_losses = 0.

            if epoch % 10 == 0:
                print(f'Now TRAINING on {epoch} EPOCHS...')

            self.train_loss[epoch-1] = self.train(X, train_labels)
            self.val_loss[epoch-1] = self.eval(X_val, test_labels)

        plt.plot(range(1, self.EPOCHS + 1), self.train_loss, 'b', label='Training loss')
        plt.plot(range(1, self.EPOCHS + 1), self.val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return (self.train_loss, self.val_loss)

    def train(self, X, labels):
        for row in range(len(X)): # SGD
            # forward
            h = self.forward(weights=self.W1, inputs=X[row], weights_size=self.hidden_size, input_size=self.input_size)
            self.train_pred[row] = self.forward(weights=self.W2, inputs=h, weights_size=self.output_size, input_size=self.hidden_size)

            # calculate loss
            self.get_loss(preds=self.train_pred[row], labels=labels[row], labels_size=self.output_size, mode='train')

            # backward
            EIP = self.backward_loss(preds=self.train_pred[row], labels=labels[row], labels_size=self.output_size)
            EI = self.backward(hidden=h, weights=self.W2, errors=EIP, hidden_size=self.hidden_size, errors_size=self.output_size)

            self.update_weights(weights=self.W2, errors=EIP, inputs=h)
            self.update_weights(weights=self.W1, errors=EI, inputs=X[row])

        return self.losses/len(X)

    def eval(self, X, labels):
        for row in range(len(X)): # SGD
            # forward
            h = self.forward(weights=self.W1, inputs=X[row], weights_size=self.hidden_size, input_size=self.input_size)
            self.val_pred[row] = self.forward(weights=self.W2, inputs=h, weights_size=self.output_size, input_size=self.hidden_size)

            # calculate loss
            self.get_loss(preds=self.val_pred[row], labels=labels[row], labels_size=self.output_size, mode='eval')

        return self.val_losses/len(X)

    def one_hot_vectorize(self, labels, true_labels):
        for i in range(len(labels)):
            if float(labels[i]) == 1.:
                true_labels[i] = 1
        return true_labels

# class NN_optimizers(object):
#     def __init__(self):
#
#     @classmethod
#     def Adagrad

# load datasets
train = np.loadtxt('./data/training.txt')
test = np.loadtxt('./data/test.txt')

print('Finished LOADING datasets.')
print()

np.random.shuffle(train)

X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

nn = NN_SGD(hidden_size=8, lr=0.01, EPOCHS=20, optimizer='SGD')
losses = nn.fit(X_train, y_train, eval_set=(X_test, y_test))

print(losses[0])
print(losses[1])
