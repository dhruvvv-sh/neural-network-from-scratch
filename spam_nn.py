"""
Spam Classifier Neural Network (NumPy Only)
Author: Your Name
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# === 1. Load and Prepare Data ===
df = pd.read_csv(
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    sep='\t', names=['label', 'message']
)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label'].values.reshape(-1, 1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Activation Functions ===
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

# === 3. Forward and Backward Propagation ===
def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, A2

def compute_cost(A2, Y):
    m = Y.shape[0]
    epsilon = 1e-8  # for numerical stability
    return -1/m * np.sum(Y * np.log(A2 + epsilon) + (1 - Y) * np.log(1 - A2 + epsilon))

def backward_pass(X, Y, Z1, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# === 4. Update Parameters ===
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2
# === 5. Training ===
input_size = x_train.shape[1]
hidden_size = 64
output_size = 1

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

learning_rate = 0.1

for i in range(1000):
    Z1, A1, A2 = forward_pass(x_train, W1, b1, W2, b2)
    cost = compute_cost(A2, y_train)
    dW1, db1, dW2, db2 = backward_pass(x_train, y_train, Z1, A1, A2, W2)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    if i % 100 == 0:
        print(f"Step {i} | Cost: {cost:.4f}")

# === 6. Evaluate Model ===
Z1_test, A1_test, A2_test = forward_pass(x_test, W1, b1, W2, b2)
predictions = (A2_test > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f"\n Test Accuracy: {accuracy * 100:.2f}%")

# === 7. Test on Custom Messages ===
def test_message_verbose(msg):
    new_input = vectorizer.transform([msg]).toarray()
    _, _, output = forward_pass(new_input, W1, b1, W2, b2)
    print(f"\nMessage: \"{msg}\"")
    print(f"Spam probability: {output[0][0]:.4f}")
    if output[0][0] > 0.5:
        print("This is SPAM")
    else:
        print("This is NOT spam")

# Try messages
test_message_verbose("Hey can we meet at 6 pm?")
test_message_verbose("Congrats! You won a free iPhone!")
