import pandas as pd
import os
import mglearn
import numpy as np

ram_prices=pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,
                                    "ram_price.csv"))
import matplotlib.pyplot as plt
plt.semilogy(ram_prices.date, ram_prices.price) #log scale
plt.show()

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

data_train=ram_prices[ram_prices.date<2000]
data_test=ram_prices[ram_prices.date>=2000]

X_train=data_train.date[:,np.newaxis]
y_train=np.log(data_train.price)

tree=DecisionTreeRegressor().fit(X_train,y_train)
linear_reg=LinearRegression().fit(X_train, y_train)

X_all=ram_prices.date[:,np.newaxis]
pred_tree=tree.predict(X_all)
pred_lr=linear_reg.predict(X_all)

price_tree=np.exp(pred_tree)
price_lr=np.exp(pred_lr)

plt.semilogy(data_train.date,data_train.price, label="training data")
plt.semilogy(data_test.date, data_test.price, label="test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear Prediction")
plt.legend()
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X,y=make_moons(n_samples=100, noise=0.25, random_state=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,stratify=y,random_state=42)

forest=RandomForestClassifier(n_estimators=5,random_state=2)
forest.fit(X_train, y_train)
print(forest.score(X_train,y_train))
print(forest.score(X_test,y_test))
print(forest.feature_importances_)









