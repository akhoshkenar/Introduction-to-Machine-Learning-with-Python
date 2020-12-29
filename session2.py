import sys
print(sys.version)

import sklearn
print(sklearn.__version__)

import numpy as np
print(np.__version__)

from sklearn.datasets import load_iris
iris_dataset=load_iris()

print(iris_dataset.keys())
#print(iris_dataset.values())
print(iris_dataset['DESCR'])
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
print(type(iris_dataset['data']))
print(iris_dataset['data'].shape)
print(iris_dataset['data'][:10])
print(iris_dataset['target'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)

print(X_train.shape)
print(y_train.shape)


import pandas as pd
import matplotlib.pyplot as plt
iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(10,10),
                           marker="o", hist_kwds={'bins':20}, alpha=0.8)
plt.show()

