import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        
    def sgd(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_epochs):
            for i in range(n_samples):
                error = y[i] - self.predict(X[i])
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                
    def bgd(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_epochs):
            error = y - self.predict(X)
            self.weights += self.learning_rate * np.dot(X.T, error) / n_samples
            self.bias += self.learning_rate * np.sum(error) / n_samples
            
    def pseudoinverse(self, X, y):
        self.weights = np.dot(np.linalg.pinv(X), y)
        self.bias = 0
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


data = pd.read_csv('reg_problem.csv') 

X = data['x'].values
y = data['y'].values

X_with_bias = np.column_stack((np.ones(len(X)), X))

adaline = Adaline(learning_rate=0.01, n_epochs=1000)
adaline.sgd(X_with_bias, y)

plt.scatter(X, y, label='Entradas')
plt.plot(X, adaline.predict(X_with_bias), label='ADALINE (sgd)',
color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresión lineal con ADALINE (sgd)')
plt.show()