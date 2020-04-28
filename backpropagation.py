
from sklearn.neural_network import MLPClassifier

#Network, that implements cell mechanic from Convay`s Game of Life.
#It receives current state of cell and amount of living neighbours and predict state of the cell at the next iteration
#Нейросеть, которая по текущему состоянию клетки и количеству живых соседей предсказывает состояние на следующей итерации

clf = MLPClassifier(solver='lbfgs', random_state=1)

clf.fit([[0, 0],
         [0, 1],
         [0, 2],
         [0, 3],
         [0, 4],
         [0, 5],
         [0, 6],
         [0, 7],
         [0, 8],
         [1, 0],
         [1, 1],
         [1, 2],
         [1, 3],
         [1, 4],
         [1, 5],
         [1, 6],
         [1, 7],
         [1, 8]],
        [0, 0, 0, 1, 0, 0, 0, 0, 0,
                   0, 0, 1, 1, 0, 0, 0, 0, 0,])

print(clf.predict([[0, 3], [0, 1], [0, 5], [1, 1], [1,3], [1,2], [1, 7]]))
