import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# load the data set
data = pd.read_csv('CTG1.csv')
data.head(3)
# print info about columns in the dataframe
#print(data)
pd.value_counts(data['aa']).plot.bar()
data['aa'].value_counts()
#plt.show()
X = np.array(data.loc[:, data.columns != 'aa'])
y = np.array(data.loc[:, data.columns == 'aa'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '3': {} \n".format(sum(y_train==3)))

sm = SMOTE(random_state=2, k_neighbors=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '3': {}".format(sum(y_train_res==3)))

y_train_res = y_train_res.reshape(-1, 1)
data = np.concatenate((X_train_res, y_train_res), axis=1)

np.savetxt("dataset3.csv", data, delimiter=",",  fmt=["%.3f" for i in range(X.shape[1])] + ["%i"])