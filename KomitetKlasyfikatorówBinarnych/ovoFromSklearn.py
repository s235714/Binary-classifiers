import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

dataset = 'CTGDaneZbalansowane'
dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
model = OneVsOneClassifier(clf, n_jobs=-1)
scores = []
for train, test in rskf.split(X, y):
    model.fit(X[train], y[train])
    y_pred = model.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("OvO accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model = OneVsOneClassifier(clf, n_jobs=-1)
scores = []
for train, test in rskf.split(X, y):
    model.fit(X[train], y[train])
    y_pred = model.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("OvO accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = DecisionTreeClassifier(criterion="gini", max_depth=5)
model = OneVsOneClassifier(clf, n_jobs=-1)
scores = []
for train, test in rskf.split(X, y):
    model.fit(X[train], y[train])
    y_pred = model.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("OvO accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
model = OneVsOneClassifier(clf, n_jobs=-1)
scores = []
for train, test in rskf.split(X, y):
    model.fit(X[train], y[train])
    y_pred = model.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("OvO accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = DecisionTreeClassifier(criterion="gini", max_depth=7)
model = OneVsOneClassifier(clf, n_jobs=-1)
scores = []
for train, test in rskf.split(X, y):
    model.fit(X[train], y[train])
    y_pred = model.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("OvO accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = DecisionTreeClassifier(criterion="entropy", max_depth=7)
model = OneVsOneClassifier(clf, n_jobs=-1)
scores = []
for train, test in rskf.split(X, y):
    model.fit(X[train], y[train])
    y_pred = model.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("OvO accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
