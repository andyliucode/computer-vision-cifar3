# Andy Liu

import numpy as np
import scipy.io
from sklearn import svm
import sklearn.model_selection

# Load the training data
training_data_path = '../handouts/data_mat/data_batch'
data = scipy.io.loadmat(training_data_path)
X, y = data['data'], data['labels']

# Load the test data
test_data_path = '../handouts/data_mat/test_data'
data = scipy.io.loadmat(test_data_path)

# Process the test data
x_test = data['data']
x_test = np.reshape(x_test, (x_test.shape[0], 3, 32, 32))
x_test = np.swapaxes(x_test, 1, 2)
x_test = np.swapaxes(x_test, 2, 3)

# Train the model
SVM_model = svm.SVC()
SVM_model.fit(X, y)

# Generate predictions
predictions = SVM_model.predict_classes(x_test)

# Save predictions to csv
np.savetxt("SVM_results.csv", predictions, delimiter=",")
