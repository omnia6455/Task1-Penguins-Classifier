import numpy as np

#Perceptron class implementation
class Perceptron:
    m=1000
    eta=0.01
    def __init__(self, learning_rate=eta, epochs=m,use_bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_bias = use_bias
        self.bias = 0.0
        self.weights = None
        self.errors = []

    def activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def fit(self, X, y):
        self.weights = np.random.uniform(-0.01, 0.01, X.shape[1])
        self.bias = 0.0
        self.errors = []
        for epoch in range(self.epochs):
            errors_this_epoch = 0
            for i in range(X.shape[0]):
                b = self.bias if self.use_bias else 0.0
                v = np.dot(self.weights, X[i]) + b
                y_pred = self.activation_function(v)
                error = y[i] - y_pred
                self.weights = self.weights + self.learning_rate * error * X[i]
                if self.use_bias:
                    self.bias = self.bias + self.learning_rate * error
                if error != 0.0:
                    errors_this_epoch += 1
            self.errors.append(errors_this_epoch)
            if errors_this_epoch == 0:
                break
        return self

    def predict(self, X):
        if self.use_bias:
            b= self.bias
        else:
            b=0.0
        v = X.dot(self.weights) + b
        return np.array([self.activation_function(val) for val in v])

    def boundary_coeffs(self):
        if self.use_bias:
            return self.weights[0], self.weights[1], self.bias
        else:
            return self.weights[0], self.weights[1], 0.0

#Adaline class implementation
class Adaline:
    def __init__(self, learning_rate=0.01, epochs=100,mse_threshold=0.01, use_bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.mse_threshold = mse_threshold
        self.use_bias = use_bias
        self.bias = 0.0
        self.mse_history = []
        self.mean = None
        self.std = None

    #Scaling to prevent Gradient Explosion and ensuring that the features are on a similar scale
    def preprocess_data(self, X):
        return (X - self.mean) / self.std

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        Z = self.preprocess_data(X)
        self.weights = np.random.uniform(-0.01, 0.01, Z.shape[1])
        self.bias = 0.0
        self.mse_history = []
        for epoch in range(self.epochs):
            for i in range(Z.shape[0]):
                b = self.bias if self.use_bias else 0.0
                v = np.dot(self.weights, Z[i]) + b
                y_pred = v
                error = y[i] - y_pred
                self.weights = self.weights + self.learning_rate * error * Z[i]
                if self.use_bias:
                    self.bias = self.bias + self.learning_rate * error
            b= self.bias if self.use_bias else 0.0
            mse = np.mean((y - (np.dot(Z, self.weights) + b)) ** 2)
            self.mse_history.append(mse)
            if mse < self.mse_threshold:
                break
        return self

    def predict(self, X):
        if self.use_bias:
            b = self.bias
        else:
            b = 0.0
        return self.preprocess_data(X).dot(self.weights) + b

    def predict_class(self, X):
        return np.where(self.predict(X) >= 0, 1, -1)

    def boundary_coeffs(self):
        w = self.weights
        W = w / self.std
        b = (self.bias if self.use_bias else 0.0) - np.sum(w * self.mean / self.std)
        return W[0], W[1], b

#Evaluate predictions function to calculate confusion matrix and accuracy
def evaluate_predictions(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == 1 and pred_val == 1:
            TP += 1
        elif true_val == -1 and pred_val == 1:
            FP += 1
        elif true_val == -1 and pred_val == -1:
            TN += 1
        elif true_val == 1 and pred_val == -1:
            FN += 1
    total = TP + TN + FP + FN
    if total > 0:
        accuracy = (TP + TN) / total
    else:
        accuracy = 0.0
    confusion_matrix = {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }
    return confusion_matrix, accuracy