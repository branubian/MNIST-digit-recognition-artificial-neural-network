import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
train = int(input("enter number of training samples to train on: "))
test = int(input("enter number of testing samples to test on: "))
train_plus_test = train + test
Y_train = data_t[0, :train]  ## code to grab first row from matrix (changed to grab 4000 columns from first row for input variable m training samples)
print(Y_train, "this is the matrix of training labels, it's shape is ", Y_train.shape)
testing_y_labels = data_t[0, train:train_plus_test]
print(testing_y_labels, "this is the testing y labels matrix, it's shape is ",testing_y_labels.shape)
X_train = data_t[1:, :train]  ## code to select all columns from second row onwards. (changed to grab upto 4000 columns for input variable m training samples)
X_train = X_train / 255  ## i don't know what this does, this isn't standard scaling either but the guy on youtube did it & it fixed the problem where softmax returned zero on every output node okay so it actually is standard scaling because if each pixel value is from 0 to 255 based on how bright the pixel is then it just converts them to a value between zero and one respectively.
X_train = X_train.T
print(X_train, "this is the training input layer divided by 255 for standard scaling, ", X_train.shape, " is it's shape")
n, m = X_train.shape
print("from the training input layer dimensions, the number of features is",n,"& the number of training samples is",m)
testing_X = data_t[1:,train:train_plus_test]
testing_X = testing_X / 255
testing_X = testing_X.T
print(testing_X, "this is the testing input layer, it's dimensions are ", testing_X.shape)


def one_hot(y):
    y_int = y.astype(int)
    num_classes = len(np.unique(y_int))
    one_hot_encoded = np.eye(num_classes)[y_int.reshape(-1)]
    return one_hot_encoded


Y_o_h_train = one_hot(Y_train)
Y_o_h_test = one_hot(testing_y_labels)

##print(one_hot(Y), "this is labels matrix one hot and it's shape is", one_hot(Y).shape)

n1 = int(input("enter number of neurons in first hidden layer (32): "))
n2 = int(input("enter number of neurons in second hidden layer (32): "))
alpha = float(input("enter floating point learning rate for gradient descent (0.01): "))
epochs = int(input("enter integer value for number of training epochs (2000): "))
n = [784, n1, n2, 10]
np.random.seed(0)
W1 = 0.01 * np.random.randn(n[0], n[1])
W2 = 0.01 * np.random.randn(n[1], n[2])
W3 = 0.01 * np.random.randn(n[2], n[3])
b1 = np.zeros((1, n[1]))
b2 = np.zeros((1, n[2]))
b3 = np.zeros((1, n[3]))


def softmax(inputs):
    # Get unnormalized probabilities
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    # Normalize them for each sample
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    output = probabilities
    return output


def loss(pred, o_h_true):  ## categorical cross entropy loss
    pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)
    true_clipped = np.clip(o_h_true, 1e-7, 1 - 1e-7)
    correct_confidences = np.sum(pred_clipped * true_clipped, axis=1)
    negative_logs = -np.log(correct_confidences)
    loss = np.mean(negative_logs)
    return loss


def forward(X,Y_o_h):
    global Z1, A1, Z2, A2, Z3, Y_hat, W1, W2, W3, b1, b2, b3, Loss
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = np.maximum(0, Z2)
    Z3 = np.dot(A2, W3) + b3
    Y_hat = softmax(Z3)  ## A3
    Loss = loss(Y_hat, Y_o_h)
    ##print(Y_hat, "this is Y_hat, it's dimensions are:", Y_hat.shape)
    ##print(Loss, "this is the categorical cross entropy loss")

def backward(X,Y):
    global dL_dW3, dL_db3, dL_dW2, dL_db2, dL_dW1, dL_db1
    samples = len(Y_hat) # counts number of rows of Y_hat
    # Copy so we can safely modify
    Y_1D = Y.ravel()
    dL_dZ3 = Y_hat.copy()
    # Calculate gradient
    dL_dZ3[range(samples), Y_1D.astype(int)] -= 1
    dL_dZ3 = dL_dZ3 / samples
    dL_dW3 = np.dot(A2.T, dL_dZ3)
    dL_db3 = np.sum(dL_dZ3, axis=0, keepdims=True)
    dL_dA2 = np.dot(dL_dZ3, W3.T)
    # Since we need to modify original variable,
    # let’s make a copy of values first
    dL_dZ2 = dL_dA2.copy()
    # Zero gradient where input values were negative
    dL_dZ2[Z2 <= 0] = 0
    dL_dW2 = np.dot(A1.T, dL_dZ2)
    dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)
    dL_dA1 = np.dot(dL_dZ2, W2.T)
    # Since we need to modify original variable,
    # let’s make a copy of values first
    dL_dZ1 = dL_dA1.copy()
    # Zero gradient where input values were negative
    dL_dZ1[Z1 <= 0] = 0
    dL_dW1 = np.dot(X.T, dL_dZ1)
    dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)

def update_params_vanilla():
    global W1, W2, W3, b1, b2, b3
    W1 = W1 - (alpha * dL_dW1)
    W2 = W2 - (alpha * dL_dW2)
    W3 = W3 - (alpha * dL_dW3)
    b1 = b1 - (alpha * dL_db1)
    b2 = b2 - (alpha * dL_db2)
    b3 = b3 - (alpha * dL_db3)

def update_params_ADAM():
    # Hyperparameters
    # learning rate
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    t = 1  # time step (increment each iteration)

    # --- Parameters ---
    W = {
        'W1': W1, 'W2': W2, 'W3': W3,
    }
    b = {
        'b1': b1, 'b2': b2, 'b3': b3,
    }

    # --- Gradients ---
    dW = {
        'W1': dL_dW1, 'W2': dL_dW2, 'W3': dL_dW3,
    }
    db = {
        'b1': dL_db1, 'b2': dL_db2, 'b3': dL_db3,
    }

    # --- Moment estimates (initialize to zeros only once) ---
    mW = {k: np.zeros_like(v) for k, v in W.items()}
    vW = {k: np.zeros_like(v) for k, v in W.items()}
    mb = {k: np.zeros_like(v) for k, v in b.items()}
    vb = {k: np.zeros_like(v) for k, v in b.items()}

    # --- Adam Update ---
    for k in W:
        # Update weights
        mW[k] = beta1 * mW[k] + (1 - beta1) * dW[k]
        vW[k] = beta2 * vW[k] + (1 - beta2) * (dW[k] ** 2)

        mW_hat = mW[k] / (1 - beta1 ** t)
        vW_hat = vW[k] / (1 - beta2 ** t)

        W[k] -= alpha * mW_hat / (np.sqrt(vW_hat) + epsilon)

    for k in b:
        # Update biases
        mb[k] = beta1 * mb[k] + (1 - beta1) * db[k]
        vb[k] = beta2 * vb[k] + (1 - beta2) * (db[k] ** 2)

        mb_hat = mb[k] / (1 - beta1 ** t)
        vb_hat = vb[k] / (1 - beta2 ** t)

        b[k] -= alpha * mb_hat / (np.sqrt(vb_hat) + epsilon)

    # Increment timestep
    t += 1

def get_accuracy(Y_hat,Y):
    global accuracy, predictions
    ##print(f"{Y_hat} This is softmax_outputs, it's dimensions are {Y_hat.shape}")
    # Target (ground-truth) labels for 3 samples
    class_targets = Y.ravel()
    class_targets = class_targets.astype(int)

    ##print(f"{class_targets} This is class_targets, it's dimensions are {class_targets.shape}")
    # Calculate values along second axis (axis of index 1)
    predictions = np.argmax(Y_hat, axis=1)

    ##print(f"{predictions} this is prediction matrix, it's dimensions: {predictions.shape}")

    accuracy = np.mean(predictions == class_targets)
    ##print(f"accuracy over training data is {accuracy*100}%")

epoch_count = []
loss_count = []
accuracy_count = []

for i in range(1,epochs+1):
    forward(X_train,Y_o_h_train)
    get_accuracy(Y_hat,Y_train)
    accuracy_count.append(float(accuracy*100))
    loss_count.append(float(Loss))
    backward(X_train,Y_train)
    update_params_ADAM()
    epoch_count.append(int(i))
    print(i,"epochs completed")




plt.subplot(2,2,1)

print(f"final loss over training is {Loss}")

print(f"final accuracy over training data is {accuracy*100}%")

plt.plot(epoch_count,loss_count)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"final loss is {Loss}")

plt.subplot(2,2,2)
plt.plot(epoch_count,accuracy_count)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (in %age)")
plt.title(f"final accuracy is {accuracy*100}%")

plt.tight_layout() ##prevents overwriting in graphs
plt.show()

forward(testing_X,Y_o_h_test)
get_accuracy(Y_hat,testing_y_labels)

print(f"final loss over testing is {Loss}")

print(f"final accuracy over testing data is {accuracy*100}%")