import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# load the data set
data = pd.read_csv('CTG10.csv')
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before Undersampling, counts of label '1': {}".format(sum(y_train == 10)))
print("Before Undersampling, counts of label '2': {} \n".format(sum(y_train == 10)))

nr = RandomUnderSampler()

X_train_miss, y_train_miss = nr.fit_resample(X_train, y_train.ravel())

print('After Undersampling, the shape of train_X: {}'.format(X_train_miss.shape))
print('After Undersampling, the shape of train_y: {} \n'.format(y_train_miss.shape))

print("After Undersampling, counts of label '1': {}".format(sum(y_train_miss == 10)))
print("After Undersampling, counts of label '2': {}".format(sum(y_train_miss == 10)))

y_train_miss = y_train_miss.reshape(-1, 1)
data = np.concatenate((X_train_miss, y_train_miss), axis=1)
np.savetxt("dataset10.csv", data, delimiter=",",  fmt=["%.3f" for i in range(X.shape[1])] + ["%i"])
