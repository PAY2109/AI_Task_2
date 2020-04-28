from sklearn import linear_model

#Neural network that acts like logical AND
#Нейронная сеть, реализующая функцию логического И
linearModel = linear_model.RidgeClassifier(alpha=0.5)
linearModel.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
print(linearModel.predict([[0, 0], [1, 0], [0, 1], [1, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1]]))
