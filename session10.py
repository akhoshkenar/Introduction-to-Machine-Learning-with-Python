from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

X_train, X_test, y_train, y_test=train_test_split(
    cancer.data, cancer.target, random_state=0)

gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))

gbrt=GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))

gbrt=GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))

print(gbrt.feature_importances_)

from sklearn.datasets import make_blobs
X,y=make_blobs(centers=4, random_state=8)

from sklearn.svm import LinearSVC
#linear_svm=LinearSVC().fit(X,y)

import numpy as np
X_new=np.hstack([X,X[:,1:]**2])

from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt

figure=plt.figure()

ax=Axes3D(figure, elev=-152, azim=-26)
mask=y==0

ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask,2], c='b',
           s=60, edgecolor='k')
ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask,2], c='r',marker='^',
           s=60, edgecolor='k')
plt.show()


