import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings("ignore")

class OneVersusOneClassifiersEnsemble(BaseEnsemble, ClassifierMixin):
    """
    OvO binary classifiers ensemble
    Komitet klasyfikatorów binarnych w schemacie OvO
    """

    def __init__(self, base_estimator=None, n_subspace_features=5, soft_voting=True, random_state=None):
        # Klasyfikator bazowy
        self.base_estimator = base_estimator
        # Liczba cech w jednej podprzestrzeni
        self.n_subspace_features = n_subspace_features
        # Tryb podejmowania decyzji
        self.soft_voting = soft_voting
        # Ustawianie ziarna losowosci
        self.random_state = random_state
        np.random.seed(self.random_state)


    def fit(self, X, y):
        # Sprawdzenie, czy X i y mają własciwy ksztalt
        X, y = check_X_y(X, y)
        # Przechowywanie  i sprawdzenie liczby klas
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.n_classes = n_classes
        # Zapis liczby atrybutow
        self.n_features = X.shape[1]
        # Utworzenie macierzy par klas
        self.pairs = []
        # Utworzenie macierzy klasyfikatorów dla każdej klasy
        self.classifiers = []
        # Utworzenie macierzy z najlepszymi cechami dla kazdej pary klas
        self.best_features = []

        #Dekompozycja zbioru na poszczególne pary
        for i in range (n_classes):
            for j in range (n_classes):
                if i < j:
                    self.pairs.append([i, j])
                    mask_i = y == i
                    mask_j = y == j
                    mask = mask_i + mask_j

                    X_ = X[mask]
                    y_ = y[mask]

                    selected_features = SelectKBest(score_func=f_classif, k=self.n_subspace_features).fit(X_, y_)
                    X__ = selected_features.transform(X_)

                    clf = clone(self.base_estimator)
                    clf.fit(X__, y_)

                    self.classifiers.append(clf)
                    self.best_features.append(selected_features)

        # Sprawdzenie, czy liczba cech w podprzestrzeni jest mniejsza od całkowitej liczby cech
        if self.n_subspace_features > self.n_features:
            raise ValueError("Number of features in subspace higher than number of features.")

        return self

    def predict(self, X):
        # Sprawdzenie, czy modele są wyuczone
        check_is_fitted(self, "classes")
        # Sprawdzenie poprawności danych
        X = check_array(X)
        # Sprawdzenie, czy liczba cech się zgadza
        if X.shape[1] != self.n_features:
            raise ValueError("Number of features does not match")
        # Utworzenie macierzy wsparć dla każdej pary klas
        predict_probas = np.zeros((X.shape[0], self.n_classes))

        for clf, selected_features, pair in zip(self.classifiers, self.best_features, self.pairs):
            vote = clf.predict_proba(selected_features.transform(X))
            vote = vote if self.soft_voting else vote > .5
            predict_probas[:, pair] += vote
            y_pred = np.argmax(predict_probas, axis=1)

        return y_pred
