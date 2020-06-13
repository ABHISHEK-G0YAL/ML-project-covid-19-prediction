import data_loader as dtl
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def process_data(raw_data):
    max_case = max(raw_data)
    max_case += max_case / 3
    raw_data = [case / max_case for case in raw_data]
    train_data = [(np.array(raw_data[i-11:i-1]).reshape(1, 10), np.array(raw_data[i]).reshape(1, 1)) for i in range(11, len(raw_data))]
    return train_data, max_case

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class layer:
    def __init__(self, inp_size, out_size, ETA):
        self.ETA = ETA
        self.b = np.random.randn(1, out_size)
        self.w = np.random.randn(inp_size, out_size)
        self.bgrad = 0.0
        self.wgrad = 0.0
    def forward(self, inp):
        self.x = inp
        self.y = np.dot(self.x, self.w) + self.b
        self.a = sigmoid(self.y)
        return self.a
    def compgrad(self, a_grad):
        y_grad = sigmoid_prime(self.y) * a_grad
        self.bgrad += y_grad
        self.wgrad += np.dot(self.x.T, y_grad)
        return np.dot(y_grad, self.w.T)
    def backprop(self):
        self.w -= self.ETA * self.wgrad
        self.b -= self.ETA * self.bgrad
        self.bgrad = 0
        self.wgrad = 0

class model:
    def __init__(self, net_size, ETA):
        self.layers = []
        for i in range(len(net_size) - 1):
            self.layers.append(layer(net_size[i], net_size[i + 1], ETA))
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    def compgrad(self, a_grad):
        for l in reversed(self.layers):
            a_grad = l.compgrad(a_grad)
    def backprop(self):
        for l in reversed(self.layers):
            l.backprop()

def train(data, BATCH_SIZE= 10):
    data_size = len(data)
    # random.shuffle(data)
    batches = (data[i : i + BATCH_SIZE] for i in range(0, data_size, BATCH_SIZE))
    loss = 0.0
    for batch in batches:
        for x, y in batch:
            a = network.forward(x)
            loss += (a - y) ** 2 / BATCH_SIZE
            a_grad = 2 * (a - y) / BATCH_SIZE
            network.compgrad(a_grad)
        network.backprop()
    return np.sum(loss) * BATCH_SIZE / data_size

def test(data, n, line='-'):
    plt.grid()
    plt.plot(raw_train_data, line)
    size = len(raw_train_data)
    out = [network.forward(x).item() for x,_ in data]
    plt.plot(list(range(11, size)), [c * max_case for c in out], line)
    pred = []
    for i in range(12):
        pred.append(network.forward(np.array(out[-10:]).reshape(1, 10)).item())
        out.append(pred[-1])
    plt.plot(list(range(size, size+12)), [c * max_case for c in pred], line)
    plt.legend(['ground_truth', 'test', 'prediction'])
    plt.savefig(f'MLP_predction{str(n)}.png')
    plt.clf()

EPOCHS = 300
BATCH_SIZE = 1
ETA = 0.04
data_loader = dtl.get_data()
raw_train_data = data_loader['totalrecovered']
train_data, max_case = process_data(raw_train_data)
# network = model([10, 100, 32, 10, 1], ETA)    # 1
# network = model([10, 100, 100, 1], ETA)       # 2
# network = model([10, 100, 32, 100, 1], ETA)   # 3
# network = model([10, 200, 130, 200, 150, 200, 130, 200, 150, 1], ETA)           # 4
network = model([10, 80, 100, 100, 120, 140, 140, 120, 100, 100, 25, 1], ETA)   # 5
test(train_data, 0, line='x')
for itr in range(EPOCHS):
    print(f'Loss: {train(train_data, BATCH_SIZE)} Epoch: {itr}')
    if itr % 100 == 99:
        test(train_data, itr, line='x')