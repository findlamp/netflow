"""
Check out the new network architecture and dataset!

Notice that the weights and biases are
generated randomly.

No need to change anything, but feel free to tweak
to test your network, play around with the epochs, batch size, etc!
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from netflow import *
from activation import *
# Load data
X_,y= fetch_openml('mnist_784', version=1, return_X_y=True)
#X_ = data['data']
#y_ = data['target']


y_ = np.zeros((y.shape[0],10))
for i in range(len(y)):
    num = int(y[i])
    y_[i][num] = 1

"""
data = load_boston()
X_ = data['data']
y_ = data['target']
"""
print(X_.shape,y_.shape)
# Normalize data

#X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 256
n_output = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, n_output)
b2_ = np.zeros(n_output)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
s2 = Sigmoid(l2)
cost = Cross_Entropy(y, s2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 300
batch_size = 128
# Total number of examples

m = X_.shape[0]

steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))


def softmax(result):
    softmax_result = np.zeros((result.shape[0],result.shape[1]))
    for i in range(len(result)):
        max_value = max(result[i])
        max_index = [j for j in range(len(result[i])) if max_value == result[i][j]]
        softmax_result[i][max_index[0]] = 1
    return softmax_result

def accurate(label,result):
    total_num = len(label)
    correct_num = 0
    label = [np.where(r==1)[0][0] for r in label]
    result =[np.where(r==1)[0][0] for r in result]

    for i in range(len(label)):
        if label[i] == result[i]:
            correct_num +=1
    return correct_num / total_num * 100

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables,0.1)

        loss += graph[-1].value
        result = graph[-2].value
    
    result = softmax(result)
    possibility = accurate(y_batch,result)
    print("Epoch: {}, Loss: {:.8f}, Accurate: {:.3f}".format(i+1, loss/steps_per_epoch,possibility))