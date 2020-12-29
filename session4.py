from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print(cancer.keys())
print(cancer.data.shape)
import numpy as np
print(np.bincount(cancer.target))
print(cancer.feature_names)
from sklearn.datasets import load_boston
boston=load_boston()
print(boston.data.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], random_state=0)
from sklearn.neighbors import KNeighborsClassifier
cancer_knn=KNeighborsClassifier(n_neighbors=3)
cancer_knn.fit(X_train, y_train)
print(cancer_knn.predict(X_test))
print(cancer_knn.score(X_test,y_test))

import matplotlib.pyplot as plt
print(cancer.feature_names)
print(cancer['data'][1])
plt.scatter(cancer['data'][1],cancer['data'][2])
plt.show()

import pandas as pd
cancer_df=pd.DataFrame(X_train, columns=cancer.feature_names)
pd.plotting.scatter_matrix(cancer_df[['mean radius','mean texture']], c=y_train)
plt.show()

pd.plotting.scatter_matrix(cancer_df.iloc[1:100,1:3])
plt.show()

from sklearn.datasets import make_blobs
import mglearn
X,y=mglearn.datasets.load_extended_boston()
print(X.shape)

X,y=mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0], X[:,1],y)
plt.legend(["a","b"], loc=4)
plt.xlabel("1st feature")
plt.ylabel("2nd feature")
plt.show()

X,y=mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'x')
plt.ylim(-3,3)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 3))


