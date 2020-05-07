# NetFlow
---

This is a simplified neural network framework to provide more people to understand how the neural network work and not only know how to use it.

---

## Activation.py 

This file includes some activation function, like relu, sigmoid, tahn, and other activation function still developing.

---

## netflow.py

This file includes some layer operations or models, like MSE and cross entropy cost funtion, and linear operation. 

---

## Minst_test.py

This file includes a example about the minst classification task.

---

## How to use this framework?

* **Linear**

```python
# Linear operation is W*X+b
# X is a feature matrix of input, and the size is m*n. 
# m is the number of tranning example, n is the number of feature
# W is a weight matrix, and the size is m*n
# m is the number of the input feature, and nn is the number of the neurons in the next layer
# b is a number or a vector, it can be broadcast by the numpy matrix, so you don't need to worry about that we can not add a number to a matrix.
layer = Linear(X, W, b)
```

* **Sigmoid**

```python
# Sigmoid function is a activation function to calculate each value in the input matrix
s = Sigmoid(layer)
```

* **Cross_Entropy**

```python
# Cross_Entropy cost function is a model to calculate the differentiate in the classification mission
# y is a label matrix from the training example, and the size is m*n
# m is the number of tranning example, n is the number of the categories, called label, the label must be one_hot label
# s is the result of computation of the neural network, and this size if m*n
# m is the number of tranning example, n is the number of the categories or the number of the neurons of the output layer

cost = Cross_Entropy(y, s)
```

* **MSE**

```python
# MSE cost function is a model to calculate the differentiate in Linear fitting mission
# y is a label vector from the training example, and the size is m*1
# m is the number of tranning example
# s is the result of computation of the neural network, and the size is m*1
# m is the number of tranning example
cost = MSE(y, s)
```

**The whole process to use this framework is in minst_test.py file, which is a minst handwritting image classification example provide for you.**





