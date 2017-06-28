# Andy Liu

import numpy as np
import scipy.io
import matplotlib.pyplot as plt 

# Load the training data
training_data_path = '../handouts/data_mat/data_batch'
data = scipy.io.loadmat(training_data_path)
X, y = data['data'], data['labels']

print(X.shape)
print(y.shape)

# Load the test data
test_data_path = '../handouts/data_mat/test_data'
data = scipy.io.loadmat(test_data_path)

# Process the test data
x_test = data['data']
# x_test = np.reshape(x_test, (x_test.shape[0], 3, 32, 32))
# x_test = np.swapaxes(x_test, 1, 2)
# x_test = np.swapaxes(x_test, 2, 3)
# x_test = np.reshape(x_test, (x_test.shape[0], 3072))

print(x_test.shape)

# Visualize the training data 
plt.imshow(X[0])
plt.show()
# Visualize the test data


