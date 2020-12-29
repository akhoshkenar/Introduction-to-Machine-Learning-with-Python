from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer=load_breast_cancer()
X_train, X_test, y_train, y_test=train_test_split(cancer.data, cancer.target,
                 stratify=cancer.target, random_state=66)

training_accuracy=[]
test_accuracy=[]
neighbors_setting=range(1,11)

from sklearn.neighbors import KNeighborsClassifier
for n_neighbors in neighbors_setting:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))

import matplotlib.pyplot as plt
plt.plot(neighbors_setting,training_accuracy,label='training_accuracy')
plt.plot(neighbors_setting,test_accuracy,label='test_accuracy')
plt.ylabel("Accuracy")
plt.xlabel('n_neighbors')
plt.legend()
plt.show()

import mglearn
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

from sklearn.neighbors import KNeighborsRegressor
X,y=mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)
print(reg.predict(X_test))
print(reg.score(X_test,y_test))

import numpy as np

##fig,axes=plt.subplots(1,3,figsize=(15,4))
##line=np.linspace(-3,3,1000).reshape(-1,1)
##for n_neighbors, ax in zip([1,3,9],axes):
##    reg=KNeighborsRegressor(n_neighbors-n_neighbors)
##    reg.fit(X_train,y_train)
##    ax.plot(line,reg.predict(line))
##    ax.plot(X_train, y_train, '^',c=mglearn.cm2(0),markersize=8)
##    ax.plot(X_test, y_test, 'v',c=mglearn.cm2(1),markersize=8)
##
##    ax.set_title(
##        "{}neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
##            n_neighbors,reg.score(X_train,y_train),
##            reg.score(X_test,y_test)))
##    ax.set_xlabel("Features")
##    ax.set_ylabel("Target")
##axes[0].legend(["Model predictions","Training data/target",
##               "Test data/target"],loc="best")
##plt.show()

mglearn.plots.plot_linear_regression_wave()
plt.show()

from sklearn.linear_model import LinearRegression
X,y=mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=42)
lr=LinearRegression().fit(X_train,y_train)
print(lr.coef_)
print(lr.intercept_)
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))

X,y=mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
lr=LinearRegression().fit(X_train,y_train)
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))

