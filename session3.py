
from sklearn.datasets import load_iris
iris_dataset=load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
print(knn.fit(X_train,y_train))

import numpy as np
X_new=np.array([[5,2.9,1,0.2]])
print(X_new.shape)

prediction=knn.predict(X_new)
print(prediction)
print(iris_dataset['target_names'][prediction])

y_pred=knn.predict(X_test)
print(np.mean(y_pred==y_test))
print(knn.score(X_test,y_test))
