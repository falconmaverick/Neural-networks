import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

# Load and preprocess the dataset
digits = load_digits()
x_data = digits.images.reshape((len(digits.images), -1)) / 16.0
y_data = digits.target.astype(int)

# One-hot encode labels
ohe = OneHotEncoder(sparse_output=False)
y_data = ohe.fit_transform(y_data.reshape(-1, 1))

# Split data into training, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Initialize neural network parameters
def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    sigma = 0.01
    W1 = sigma * np.random.randn(hidden1_size, input_size)
    b1 = np.zeros((hidden1_size, 1))
    W2 = sigma * np.random.randn(hidden2_size, hidden1_size)
    b2 = np.zeros((hidden2_size, 1))
    W3 = sigma * np.random.randn(output_size, hidden2_size)
    b3 = np.zeros((output_size, 1))
    return W1, b1, W2, b2, W3, b3

# Activation functions
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Compute loss
def compute_loss(A3, Y):
    batch_size = Y.shape[1]
    loss = -np.sum(Y * np.log(A3 + 1e-8)) / batch_size
    return loss

# Backpropagation
def back_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3):
    m = X.shape[1]
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dZ2 = np.dot(W3.T, dZ3) * (Z2 > 0)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2, dW3, db3

# Update parameters
def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W2, b2, W3, b3

# Training loop with validation
def train_neural_network(x_train, y_train, x_val, y_val, epochs=10, learning_rate=0.01):
    input_size = x_train.shape[1]
    hidden1_size = 128
    hidden2_size = 64
    output_size = 10
    
    W1, b1, W2, b2, W3, b3 = initialize_parameters(input_size, hidden1_size, hidden2_size, output_size)
    
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(x_train.T, W1, b1, W2, b2, W3, b3)
        loss = compute_loss(A3, y_train.T)
        dW1, db1, dW2, db2, dW3, db3 = back_propagation(x_train.T, y_train.T, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3)
        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
        
        # Compute validation loss
        _, _, _, _, _, A3_val = forward_propagation(x_val.T, W1, b1, W2, b2, W3, b3)
        val_loss = compute_loss(A3_val, y_val.T)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Training Loss = {loss:.4f}, Validation Loss = {val_loss:.4f}")
    return W1, b1, W2, b2, W3, b3

# Train model with validation
W1, b1, W2, b2, W3, b3 = train_neural_network(x_train, y_train, x_val, y_val)

# Evaluate on test set
_, _, _, _, _, A3_test = forward_propagation(x_test.T, W1, b1, W2, b2, W3, b3)
predictions = np.argmax(A3_test, axis=0)
labels = np.argmax(y_test.T, axis=0)
accuracy = np.mean(predictions == labels)
print(f'Final Test Accuracy: {accuracy * 100:.2f}%')
