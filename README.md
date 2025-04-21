# MNIST digit recognition artificial neural network from scratch

Artificial neural network built from scratch without using pytorch, keras or other machine learning libraries to better gain comprehension of the math behind machine learning.

Contains two neural networks built directly from scratch, both trained on the MNIST dataset.

User can customize and decide the number of neurons in each hidden layer of both a single layer neural network and a multi layer neural network.

For training and testing accuracy measurement, it is recommended to train on 2000 to 5000 training samples with about 1000 epochs and an approx learning rate of 0.1 for backpropagation.

Run "MNIST 784xhx10.py" or "MNIST Trainable and Testable 784xh1xh2xh3x10.py" and input the desired training parameters to return accuracy on testing samples, same with the other file.

Program slices the MNIST training dataset in "train.csv" into a batch for training and a batch for testing.
