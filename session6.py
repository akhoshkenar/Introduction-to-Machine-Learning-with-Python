import mglearn
from sklearn.model_selection import train_test_split

X,y=mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
from sklearn.linear_model import Ridge
ridge=Ridge().fit(X_train,y_train)
print(ridge.score(X_train, y_train))
print(ridge.score(X_test, y_test))
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print(ridge10.score(X_train, y_train))
print(ridge10.score(X_test, y_test))
ridge01 = Ridge(alpha=0.10).fit(X_train, y_train)
print(ridge01.score(X_train, y_train))
print(ridge01.score(X_test, y_test))

from sklearn.linear_model import Lasso
lasso=Lasso().fit(X_train,y_train)
print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))

import numpy as np
print(np.sum(lasso.coef_!=0))

Lasso001=Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train)
print(Lasso001.score(X_train, y_train))
print(Lasso001.score(X_test, y_test))
print(np.sum(Lasso001.coef_!=0))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X,y=make_blobs()

#fig, axes = plt.subplots(1,2,figsize=(10,3))
#for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#    clf=model.fit(X,y)
#    mglearn.plots.plot_2d_separator(clf, X,fill=False, eps=0.5,ax=ax, alpha=0.7)
#    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#    ax.set_title("{}".format(clf.__class__.__name__))
#    ax.set_xlabel("Feature 0")
#    ax.set_ylabel("Feature 1")
#axes[0].legend()
#plt.show()
