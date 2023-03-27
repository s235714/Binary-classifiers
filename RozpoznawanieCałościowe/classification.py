import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import clone
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn.metrics as skm
from scipy.stats import ttest_ind
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
sns.set_theme()

classes = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10"
}

columns = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '20',
    '21',
    'Klasa'
]
data = pd.read_csv('CTGDaneZbalansowane.txt', sep=';', header=None)
data.columns = columns


def classify(x, y, classifiers):
    scores = np.zeros((len(classifiers), len(columns) - 1, 10))
    maxIterations = len(classifiers) * (len(columns) - 1) * 10
    currentIteration = 0
    for clfId, clfName in enumerate(classifiers):
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1234)
        for featuresCount in range(1, len(columns)):
            selector = SelectKBest(score_func=chi2, k=featuresCount)
            selectedFeatures = selector.fit_transform(x, y)
            for foldId, (trainIndex, testIndex) in enumerate(rskf.split(selectedFeatures, y)):
                xTrain, xTest = selectedFeatures[trainIndex], selectedFeatures[testIndex]
                yTrain, yTest = y[trainIndex], y[testIndex]
                DecTree = clone(classifiers[clfName])
                DecTree.fit(xTrain, yTrain)
                predict = DecTree.predict(xTest)
                scores[clfId, featuresCount - 1, foldId] = skm.accuracy_score(yTest, predict)
                currentIteration += 1
                print("%d / %d" % (currentIteration, maxIterations))
    np.save('resultsChi2', scores)

def analyze(classifiers):
    scores = np.load('resultsChi2.npy')
    mean = np.mean(scores, axis=2)
    std = np.std(scores, axis=2)
    for clfId, clfName in enumerate(classifiers):
        print('\n\nKlasyfikator: %s\n' % (clfName))
        for featureCount in range(1, len(columns)):
            currentMean = mean[clfId, featureCount - 1]
            currentSTD = std[clfId, featureCount - 1]
            print("Liczba cech: %d, Średnia wartość: %.3f, Odch. stand. %.2f" % (featureCount, currentMean, currentSTD))

    alfa = .05
    tStatisticArray = np.zeros((len(columns) - 1, len(classifiers), len(classifiers)))
    pValueArray = np.zeros((len(columns) - 1, len(classifiers), len(classifiers)))

    for featureIndex in range(len(columns) - 1):
        for i in range(len(classifiers)):
            for j in range(len(classifiers)):
                tStatisticArray[featureIndex, i, j], pValueArray[featureIndex, i, j] = ttest_ind(scores[i, featureIndex], scores[j, featureIndex])

    headers = ["3 Gini", "5 Gini", "7 Gini", "3 Entropy", "5 Entropy", "7 Entropy"]
    namesColumn = np.array([["3 Gini"], ["5 Gini"], ["7 Gini"], ["3 Entropy"], ["5 Entropy"], ["7 Entropy"]])

    advantage = np.zeros((len(columns) - 1, len(classifiers), len(classifiers)))
    for featureIndex in range(len(columns) - 1):
        advantage[featureIndex][tStatisticArray[featureIndex] > 0] = 1

    significance = np.zeros((len(columns) - 1, len(classifiers), len(classifiers)))
    for featureIndex in range(len(columns) - 1):
        significance[featureIndex][pValueArray[featureIndex] <= alfa] = 1

    statisticallyBetterTable = []
    for featureIndex in range(len(columns) - 1):
        statisticallyBetter = significance[featureIndex] * advantage[featureIndex]
        statisticallyBetterTable.append(tabulate(np.concatenate((namesColumn, statisticallyBetter), axis=1), headers))

    for featureIndex in range(len(columns) - 1):
        print("\nStatystycznie znacząco lepsze dla %d cech(y):\n" % (featureIndex + 1), statisticallyBetterTable[featureIndex])

def createRankingForPlot(x, y, score_func):
    selector = SelectKBest(score_func=score_func, k='all')
    selector.fit(x, y)
    ranking = [
        (name, round(score, 2))
        for name, score in zip(x.columns, selector.scores_)
    ]
    ranking.sort(reverse=True, key=lambda x: x[1])
    return ranking

def createRankingPlot(ranking):
    plt.figure(figsize=(30, 20))
    sortedRanking = sorted([(f[0], f[1]) for f in ranking], key=lambda f: f[1])
    plt.barh(range(len(ranking)), [feature[1] for feature in sortedRanking], align='center')
    plt.title("Selekcja cech metodą chi-kwadrat", fontsize=24)
    plt.xlabel('Wartość współczynnika chi-kwadrat', fontsize=16)
    plt.ylabel('Numer cechy', fontsize=16)
    plt.yticks(range(len(ranking)), [feature[0] for feature in sortedRanking])
    plt.show()

def createResultPlot(classifiers, TreeDepth):
    scores = np.load('resultsChi2.npy')
    mean = np.mean(scores, axis=2) * 100
    feature_range = np.arange(1, len(columns))
    plt.figure(figsize=(30, 20))
    for clfId, clfName in enumerate(classifiers):
        if clfName.endswith(TreeDepth):
            line = gaussian_filter(mean[clfId], sigma=1)
            plt.plot(feature_range, line, label=clfName)

    axes = plt.gca()
    axes.set_xlim([1, 22])
    axes.set_ylim([0, 100])
    x_ticks = np.arange(1, 22, 1)
    y_ticks = np.arange(0, 100, 5)
    plt.xticks(x_ticks, fontsize=24)
    plt.yticks(y_ticks, fontsize=24)
    plt.xlabel('Liczba cech', fontsize=20)
    plt.ylabel('Jakość klasyfikacji (metryka accuracy [%])', fontsize=14)
    plt.legend(fontsize=18)
    plt.title(f"Jakość klasyfikacji dla kryterium podziału drzewa {TreeDepth}", fontsize=32)
    plt.show()


def main():
    x = data.drop('Klasa', axis=1)
    y = data['Klasa']
    classifiers = {
        '3 Gini': DecisionTreeClassifier(criterion="gini", max_depth=3),
        '5 Gini': DecisionTreeClassifier(criterion="gini", max_depth=5),
        '7 Gini': DecisionTreeClassifier(criterion="gini", max_depth=7),
        '3 Entropy': DecisionTreeClassifier(criterion="entropy", max_depth=3),
        '5 Entropy': DecisionTreeClassifier(criterion="entropy", max_depth=5),
        '7 Entropy': DecisionTreeClassifier(criterion="entropy", max_depth=7),
    }
    #classify(x, y, classifiers)
    ranking = createRankingForPlot(x, y, chi2)
    createRankingPlot(ranking)
    analyze(classifiers)
    createResultPlot(classifiers, 'Gini')
    createResultPlot(classifiers, 'Entropy')

if __name__ == '__main__':
    main()
