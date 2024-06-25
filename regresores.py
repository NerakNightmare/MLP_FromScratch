import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

np.random.seed(42)
m = 300
ruido = 0.5
x = 6 * np.random.rand(m,1) - 3
y = 0.5*x**2+x+2 + ruido * np.random.randn(m,1)

xtrain, xtest, ytrain, ytest = train_test_split(x,y)


model = DecisionTreeRegressor()
#model = KNeighborsRegressor()
#model = SVR()
#model = KernelRidge(kernel='rbf', alpha=0.1)
#model = MLPRegressor()
model.fit(xtrain,ytrain)
print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))
plt.title('Decision Tree')

#Dibujar
xnew = np.linspace(-3,3,200).reshape(-1,1)
ynew = model.predict(xnew)
plt.plot(xnew,ynew, '-k', linewidth=3)
plt.plot(xtrain, ytrain, '.b')
plt.plot(xtest, ytest, '.r')
plt.xlabel(r'$x$', fontsize=15)
plt.ylabel(r'$y$', fontsize=15)
plt.axis([-3,3,0,10])
plt.show()