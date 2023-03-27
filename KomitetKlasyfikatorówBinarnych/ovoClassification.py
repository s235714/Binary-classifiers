import numpy as np
from ovo import OneVersusOneClassifiersEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

dataset = 'CTGDaneZbalansowane'
dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
print("Total number of features", X.shape[1])

n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

# Eksperyment sprawdzający wpływ trybu głosowania

clf = OneVersusOneClassifiersEnsemble(base_estimator=MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=10000), n_subspace_features=5, soft_voting=True, random_state=1234)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("Soft voting gini 7 treedepth 5 features- accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = OneVersusOneClassifiersEnsemble(base_estimator=MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=10000), n_subspace_features=5, soft_voting=False, random_state=1234)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("Hard voting gini 7 treedepth 5 features- accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = OneVersusOneClassifiersEnsemble(base_estimator=MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=10000), n_subspace_features=10, soft_voting=True, random_state=1234)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("Soft voting gini 7 treedepth 10 features- accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = OneVersusOneClassifiersEnsemble(base_estimator=MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=10000), n_subspace_features=10, soft_voting=False, random_state=1234)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("Hard voting gini 7 treedepth 10 features- accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = OneVersusOneClassifiersEnsemble(base_estimator=MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=10000), n_subspace_features=15, soft_voting=True, random_state=1234)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("Soft voting gini 7 treedepth 15 features- accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = OneVersusOneClassifiersEnsemble(base_estimator=MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=10000), n_subspace_features=15, soft_voting=False, random_state=1234)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("Hard voting gini 7 treedepth 15 features- accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = OneVersusOneClassifiersEnsemble(base_estimator=MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=10000), n_subspace_features=20, soft_voting=True, random_state=1234)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("Soft voting gini 7 treedepth 20 features- accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = OneVersusOneClassifiersEnsemble(base_estimator=MLPClassifier(hidden_layer_sizes=100, solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=10000), n_subspace_features=20, soft_voting=False, random_state=1234)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
np.save('results', scores)
print("Hard voting gini 7 treedepth 20 features- accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
