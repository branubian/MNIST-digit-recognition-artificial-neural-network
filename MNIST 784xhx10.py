import numpy as np
import pandas as pd
## import matplotlib.pyplot as plt

# Load the MNIST dataset
data = pd.read_csv("train.csv") ## reads file stored in same folder as this code & makes it a dataframe.
print(data.head(8))
data_matrix = np.array(data) ## makes matrix out of dataframe.
print(data_matrix.shape, "matrix dimensions before transposing")
print(data_matrix, "matrix before transposing, first column is all labels")
data_t = data_matrix.T
print(data_t.shape, "shape of transposed input matrix, this is the shape of data_t")
print(data_t, "transposed input matrix, now the first row is all labels, this is data_t")
n, m = data_t.shape  # n is number of features, m is number of training samples WRONG
## print("number of features is ",n," & the number of training samples is ",m)
## plt.matshow(data_t)
## plt.show() don't need to see it plotted out anymore
print("raw input data layer has ", n, " rows & ", m, " columns")
m = int(input("enter number of training samples to train on: "))
t = int(input("enter number of testing samples to test on: "))
m_plus_t = m + t
y_labels = data_t[0, :m]  ## code to grab first row from matrix (changed to grab 4000 columns from first row for input variable m training samples)
print(y_labels, "this is the matrix of labels, it's shape is ", y_labels.shape)
testing_y_labels = data_t[0, m:m_plus_t]
print(testing_y_labels, "this is the testing y labels matrix, it's shape is ",testing_y_labels.shape)
X0 = data_t[1:, :m]  ## code to select all columns from second row onwards. (changed to grab upto 4000 columns for input variable m training samples)
X0 = X0 / 255  ## i don't know what this does, this isn't standard scaling either but the guy on youtube did it & it fixed the problem where softmax returned zero on every output node okay so it actually is standard scaling because if each pixel value is from 0 to 255 based on how bright the pixel is then it just converts them to a value between zero and one respectively.
print(X0, "this is the input layer divided by 255 for standard scaling, ", X0.shape, " is it's shape")
n, m = X0.shape
print("from the input layer dimensions, the number of features is ",n," & the number of training samples is ",m)
testing_X0 = data_t[1:,m:m_plus_t]
print(testing_X0, "this is the testing input layer, it's dimensions are ", testing_X0.shape)


def one_hot_encode(labels):  ## I'll take care to understand this part of the code later but for now it does what it should
    # Get the number of classes (max label value + 1)
    num_classes = len(np.unique(labels))

    # Initialize a matrix of zeros with shape (len(labels), num_classes)
    one_hot_encoded = np.zeros((len(labels), num_classes), dtype=int)

    # Set the appropriate index to 1 for each label
    one_hot_encoded[np.arange(len(labels)), labels] = 1

    return one_hot_encoded.T

# actually one hot encode training labels
y_labels_one_hot = one_hot_encode(y_labels)
print(y_labels, "this is the matrix of training dataset labels, it's shape ", y_labels.shape)
print(y_labels_one_hot, "this is the training dataset labels actually one hot encoded, it's shape ",
      y_labels_one_hot.shape)
print(y_labels_one_hot[:, :3], "all rows from first 3 columns of above matrix")

print("asserting if number of training samples or columns is same in training input layer matrix and training label matrix")
assert X0.shape[1]==y_labels_one_hot.shape[1]
print("number of training samples or columns is same in both input layer matrix and training labels matrix.")

## now initialize weights and biases.
## n0 = 784  ## input layer
## n1 = 10  ## hidden layer sigmoid
## n2 = 11  ## hidden layer sigmoid
## n3 = 10  ## output layer sigmoid, can only have as many nodes as the classes that exist

h = int(input("enter number of neurons in our three layer neural network's hidden layer: "))

n = [784, h, 10]

W1 = np.random.randn(n[1], n[0])
b1 = np.random.randn(n[1], 1)
W2 = np.random.randn(n[2], n[1])
b2 = np.random.randn(n[2], 1)

## def ReLU(Z):
    ## return np.maximum(Z, 0)


## def softmax(Z):
    ## A = np.exp(Z) / sum(np.exp(Z))
    ## return A  ## this gives runtime error encountered in overflow in exp, add and invalid value encountered in divide.


## def softmax(Z):
## Z_max = np.max(Z)  # Find the maximum value in Z
## A = np.exp(Z - Z_max) / np.sum(np.exp(Z - Z_max))  # Subtract max(Z) before exponentiation
## return A this one just returns zero on everything



## def feedforward():
    ## global Z1, A1, Z2, A2, y_hat
    ## Z1 = W1.dot(X0) + b1
    ## A1 = ReLU(Z1)
    ## Z2 = W2.dot(A1) + b2
    ## A2 = softmax(Z2)
    ## y_hat = A2

## now we need a function that is gonna look at our y_hat matrix and put 1 to the max value in every column and put 0 to every other value in that column.
def one_hot_columns(matrix):
    result = np.zeros_like(matrix)  # Initialize a zero matrix of the same shape
    max_indices = np.argmax(matrix, axis=0)  # Get indices of max values in each column

    # Assign 1 to the max values in each column
    for col, row in enumerate(max_indices):
        result[row, col] = 1
    int_result=result.astype(int) #make the result matrix of integer values

    return int_result


def un_one_hot_columns(matrix):
    # Get the indices of the max values in each column
    max_indices = np.argmax(matrix, axis=0)

    # Reshape into a column vector (n_columns x 1)
    unflattened_result = max_indices.reshape(-1, 1).T
    return unflattened_result.flatten()


def feedforward(): ## uses sigmoid on every layer
  global Z1, A1, Z2, A2, Z3, A3, Z4, A4, y_hat, un_one_hot_y_hat
  Z1 = W1 @ X0 + b1
  A1 = 1 / (1 + np.exp(-1 * Z1))
  Z2 = W2 @ A1 + b2
  A2 = 1 / (1 + np.exp(-1 * Z2))
  y_hat = one_hot_columns(A2)
  un_one_hot_y_hat=un_one_hot_columns(y_hat)

feedforward()
print(y_hat[:, :3], "after running one epoch feed forward, this is the first three columns of one hot encoded prediction")


## def backprop_and_update_params(alpha):
    ## global W1, W2, W3, b1, b2, b3, Z3, A3, Z2, A2, Z1, A1, X0, y_hat, y_labels_one_hot, n, m

    ## dC_dZ2 = (1 / m) * (A2 - y_labels_one_hot)  ## ultimate layer propagator
    ## assert dC_dZ2.shape == (n2, 42000)

    ## dZ2_dW2 = A1
    ## dC_dW2 = dC_dZ2 @ dZ2_dW2.T  ## ultimate layer's weights
    ## assert dC_dW2.shape == (n2, n1)

    ## C_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)  ## ultimate layer's biases
    ## assert dC_db2.shape == (n2, 1)

    ## dZ2_dA1 = W2
    ## dA1_dZ1 = A1 * (1 - A1)
    ## dC_dZ1 = (dZ2_dA1.T @ dC_dZ2) * dA1_dZ1  ## penultimate layer's propagator
    ## assert dC_dZ1.shape == (n1, 42000)

    ## dZ1_dW1 = X0
    ## dC_dW1 = dC_dZ1 @ dZ1_dW1.T  ## penultimate layer's weights
    ## assert dC_dW1.shape == (n1, n0)

    ## dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)  ## penultimate layer's biases
    ## assert dC_db1.shape == (n1, 1)

    ## W2 = W2 - (alpha * dC_dW2)
    ## W1 = W1 - (alpha * dC_dW1)

    ## b2 = b2 - (alpha * dC_db2)
    ## b1 = b1 - (alpha * dC_db1)

def backprop_n_update_params(alpha):
  global W1, W2, W3, W4, b1, b2, b3, b4, Z4, A4, Z3, A3, Z2, A2, Z1, A1, X0, y_hat, y_labels_one_hot, m, n

  dC_dZ2 = (1 / m) * (A2 - y_labels_one_hot)
  assert dC_dZ2.shape == (n[2], m) ## last layer propagator

  dZ2_dW2 = A1
  dC_dW2 = dC_dZ2 @ dZ2_dW2.T
  assert dC_dW2.shape == (n[2], n[1])

  dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)
  assert dC_db2.shape == (n[2], 1)

  ## dC_dZ3 = (1/m) * (A3 - y_labels_one_hot)
  ## print("dC_dZ3 shape is ", dC_dZ3.shape)
  ## print("n[3] is ", n[3])
  ## print("m is", m)
  ## assert dC_dZ3.shape == (n[3], m)

  dZ2_dA1 = W2
  dA1_dZ1 = A1 * (1 - A1)
  dC_dZ1 = (dZ2_dA1.T @ dC_dZ2) * dA1_dZ1
  assert dC_dZ1.shape == (n[1], m) ## first layer's propagator

  dZ1_dW1 = X0
  dC_dW1 = dC_dZ1 @ dZ1_dW1.T
  assert dC_dW1.shape == (n[1], n[0])

  dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)
  assert dC_db1.shape == (n[1], 1)

  W2 = W2 - (alpha * dC_dW2)
  W1 = W1 - (alpha * dC_dW1)

  b2 = b2 - (alpha * dC_db2)
  b1 = b1 - (alpha * dC_db1)

def train(epochs, alpha):
    print("alpha is equal to ", alpha)
    for i in range(1,epochs+1):
        feedforward()
        backprop_n_update_params(alpha)
        print("epochs completed: ",i)
    print("model trained")

epochs=int(input("enter integer epochs: "))
alpha=float(input("enter floating point number alpha learning rate: "))
disp_columns=int(input("enter integer first n columns to display after training model: "))
train(epochs,alpha)
print(y_labels_one_hot[:,:disp_columns], "this is the first ", disp_columns," columns of the label matrix")
print(y_hat[:,:disp_columns], "this is the first ", disp_columns," columns of y hat after training")
print(y_labels[:disp_columns], "this is the first ", disp_columns," columns of un-one-hot y labels for training")
print(un_one_hot_y_hat[:disp_columns], "this is the first ", disp_columns," columns of un-one-hot y hat after training")

right_predictions=0
wrong_predictions=0
total_predictions=len(un_one_hot_y_hat)
##for x,y in y_labels,un_one_hot_y_hat:
        ##print("x is ", x, "y is ", y)
        ##if x == y:
            ##right_predictions =+ 1
            ##total_predictions =+ 1
            ##print("correct")
        ##if x != y:
            ##wrong_predictions =+ 1
            ##total_predictions =+ 1
            ##print("wrong")

def check_elementwise_equality(arr1, arr2):
    """Checks if values at the same index in two arrays are equal."""
    if arr1.shape != arr2.shape:
        raise ValueError("Both arrays must have the same shape")

    return arr1 == arr2  # Element-wise comparison


accuracy_array = check_elementwise_equality(y_labels,un_one_hot_y_hat)
print(accuracy_array[:disp_columns], "this is the first ", disp_columns," columns of accuracy array after training")
for i in accuracy_array:
    if i == True:
        right_predictions = right_predictions + 1
    if i == False:
        wrong_predictions = wrong_predictions + 1

print("right predictions are ", right_predictions)
print("wrong predictions are ", wrong_predictions)
print("total predictions are ", total_predictions)
percentage_accuracy = (right_predictions / total_predictions) * 100

print("percentage accuracy over training data is ", percentage_accuracy,"%")
print("training batch size (number of training samples) is: ", X0.shape[1])

print("commencing testing")

X0 = testing_X0
X0 = X0 /255
y_labels = testing_y_labels
y_labels_one_hot = one_hot_encode(y_labels)

print("asserting if number of testing samples or columns is same in testing input layer matrix and testing label matrix")
assert X0.shape[1]==y_labels_one_hot.shape[1]
print("number of testing samples or columns is same in both input layer matrix and testing labels matrix.")

print("running one epoch feedforward...")
feedforward()

print(y_hat, "this is the one hot encoded prediction matrix over testing samples.")
print(un_one_hot_y_hat, "this is prediction matrix over testing samples without one hot encoding.")


right_predictions=0
wrong_predictions=0
total_predictions=len(un_one_hot_y_hat)


accuracy_array = check_elementwise_equality(y_labels,un_one_hot_y_hat)
print(accuracy_array[:disp_columns], "this is the first ", disp_columns," columns of accuracy array after testing")
for i in accuracy_array:
    if i == True:
        right_predictions = right_predictions + 1
    if i == False:
        wrong_predictions = wrong_predictions + 1

print("right predictions are ", right_predictions)
print("wrong predictions are ", wrong_predictions)
print("total predictions are ", total_predictions)
percentage_accuracy = (right_predictions / total_predictions) * 100

print("percentage accuracy over testing data is ", percentage_accuracy,"%")


exit=input("program finished, give any input to exit")

