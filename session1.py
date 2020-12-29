import numpy as np
x=np.array([[1,2,3],[4,5,6]])
print("x:\n{}".format(x))

from scipy import sparse

x=np.eye(4)
print("x:\n{}".format(x))
print("x:\n{}".format(sparse.csr_matrix(x)))

x=np.ones(4)
row_ind=np.arange(4)
col_ind=np.arange(4)
x1=sparse.coo_matrix((x,(row_ind,col_ind)))
print("x:\n{}".format(x1))


import matplotlib.pyplot as plt
x=np.linspace(-10,10,1000)
y=np.sin(x)
plt.plot(x,y,marker="x")
plt.show()

import pandas as pd
from IPython.display import display
data={"amin":[1,2,3],"khoshkenar":["a","b","c"]}
data_pandas=pd.DataFrame(data)
display(data_pandas)
data_pandas[data_pandas.amin>1] 
