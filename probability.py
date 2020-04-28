from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#Network that uses Gaussian Classifier to act as 1-to-4 demultiplexer
#X is choice input, y is output line
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [1, 2, 3, 4]
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,
                                         random_state=0).fit(X, y)

print(gpc.predict([[1, 0], [0, 0], [0, 1], [1, 1]]))