from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression().fit(X_train,y_train)
print(logreg.score(X_train,y_train))
print(logreg.score(X_test,y_test))

logreg100=LogisticRegression(C=100).fit(X_train,y_train)
print(logreg100.score(X_train,y_train))
print(logreg100.score(X_test,y_test))

logreg001=LogisticRegression(C=0.01).fit(X_train,y_train)
print(logreg001.score(X_train,y_train))
print(logreg001.score(X_test,y_test))

logreg001L1=LogisticRegression(C=0.01, penalty="l1").fit(X_train,y_train)
print(logreg001L1.score(X_train,y_train))
print(logreg001L1.score(X_test,y_test))

import matplotlib.pyplot as plt

for C, marker in zip([0.001,1,100],['o','^','v']):
    lr_l1=LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={} is {}".format(C,lr_l1.score(X_train,y_train)))
    print("Test accuracy of l1 logreg with C={} is {}".format(C,lr_l1.score(X_test,y_test)))
    plt.plot(lr_l1.coef_.T, marker,label="{}".format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.show()

import mglearn
from sklearn.datasets import make_blobs

X,y=make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.show()

from sklearn.svm import LinearSVC
linear_svm=LinearSVC().fit(X,y)
print(linear_svm.coef_.shape)
print(linear_svm.intercept_.shape)


mglearn.discrete_scatter(X[:,0],X[:,1],y)
import numpy as np
line=np.linspace(-15,15)
for coef, intercept in zip(linear_svm.coef_, linear_svm.intercept_):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1])
plt.show()








