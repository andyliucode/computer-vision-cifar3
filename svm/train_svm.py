# Andy Liu

import scipy.io
from sklearn import svm

data = scipy.io.loadmat('handouts\data_mat\data_batch')
X, y = data['data'], data['labels']

SVM_model = svm.SVC()
SVM_model.fit(X, y)

