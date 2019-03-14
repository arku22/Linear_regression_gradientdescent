# Linear regression gradient descent
# Problem part 2

import numpy as np

# Load data
training_images = np.load("/home/archit/Desktop/smile_data/trainingFaces.npy")
training_labels = np.load("/home/archit/Desktop/smile_data/trainingLabels.npy")
test_images = np.load("/home/archit/Desktop/smile_data/testingFaces.npy")
test_labels = np.load("/home/archit/Desktop/smile_data/testingLabels.npy")

# Design matrix
X = training_images
X = np.hstack((np.ones((X.shape[0], 1)), X))    # appending bias unit to all
# Class labels
y = training_labels
y = y.reshape(y.shape[0],1)

# Assign values to notations
m = X.shape[0]  # no of training examples
n = X.shape[1]  # no of input features per image 

# Intialize parameters
theta = np.zeros((n, 1))    # weights
alpha = 9E-1    # Learning rate
tol = 0.001 # tolerance to decide no. of weight update iterations required
J_new = 0


while True:
    hypothesis = np.dot(X, theta)
    brackets = np.sum( (hypothesis - y)*X )
    #brackets = brackets.reshape(brackets.shape[0],1)
    theta = theta - alpha * (1/m) * brackets
    brackets_J = np.sum( (hypothesis - y)**2 )
    J = (1/(2*m)) * (brackets_J)
    J_old = J_new
    J_new = J
    J_diff = np.absolute(J_old - J_new)
    #print(J_diff)
    if J_diff <= tol:
        break

print("Unregularized cost for training data = " + str(J))
    
# Computing unregularized cost function on testing set

X_t = test_images     # load test images
X_t = np.hstack((np.ones((X_t.shape[0], 1)), X_t))    # appending bias unit to all
y_t = test_labels     # load test labels
y_t = y_t.reshape(y_t.shape[0], 1)

m_t = X_t.shape[0]      # number of training examples
hypothesis = np.dot(X_t, theta)
J_unreg_test = (1/(2*m_t)) * (np.sum((hypothesis - y_t)**2))
print("\nUnregularized cost for test data = " + str(J_unreg_test))
        
    
        
      
