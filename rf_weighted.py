# Random Forest 
import numpy as np
from scipy import stats
import random

# Análise dos Datasets - Métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

# Datasets 
import pandas as pd

# Train-test split
from sklearn.model_selection import train_test_split

# Calculate class weights
from collections import Counter


from scipy.stats import entropy

def calculate_class_weights(y):

    class_count = Counter(y)
    total_samples = sum(class_count.values())
    class_weights = {cls: total_samples / class_count[cls] for cls in sorted(class_count)}
    return class_weights


def weighted_entropy(p, class_weights):
   
    class_counts = np.bincount(p)
    class_probs = class_counts / np.sum(class_counts)

    # Apply weights to class probabilities
    weighted_probs = np.array([class_probs[i] * class_weights.get(i, 1) for i in range(len(class_probs))])
    weighted_probs /= np.sum(weighted_probs)  # Normalize weighted probabilities
    
    # Calculate entropy
    return entropy(weighted_probs)



def information_gain(y, splits, class_weights):
    total_weighted_entropy = weighted_entropy(y, class_weights)
    splits_entropy = sum([weighted_entropy(split, class_weights) * (len(split) / len(y)) for split in splits])
    return total_weighted_entropy - splits_entropy



def mse_criterion(y, splits):
    y_mean = np.mean(y)
    return -sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])


def xgb_criterion(y, left, right, loss):
    left = loss.gain(left["actual"], left["y_pred"])
    right = loss.gain(right["actual"], right["y_pred"])
    initial = loss.gain(y["actual"], y["y_pred"])
    gain = left + right - initial
    return gain


def get_split_mask(X, column, value):
    left_mask = X[:, column] < value
    right_mask = X[:, column] >= value
    return left_mask, right_mask


def split(X, y, value):
    left_mask = X < value
    right_mask = X >= value
    return y[left_mask], y[right_mask]


def split_dataset(X, target, column, value, return_X=True):
    left_mask, right_mask = get_split_mask(X, column, value)

    left, right = {}, {}
    for key in target.keys():
        left[key] = target[key][left_mask]
        right[key] = target[key][right_mask]

    if return_X:
        left_X, right_X = X[left_mask], X[right_mask]
        return left_X, right_X, left, right
    else:
        return left, right
    


class WeightedTree(object):
    # Implementação recursiva da Decision Tree

    def __init__(self, regression=False, criterion=None, n_classes=None):
        self.regression = regression # True se estivermos a tratar regressão (não irá ser usado)
        self.impurity = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.criterion = criterion # Criterio usado nos splits das árvores
        self.loss = None
        self.n_classes = n_classes # Usado para classificação
        self.left_child = None
        self.right_child = None

    @property
    def is_terminal(self): 

        '''Retorna True se o nó atual não tiver filhos, ou seja, for uma folha'''

        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X):

        '''Encontra todos os valores de split possíveis'''

        split_values = set() # Conjunto dos possíveis split values - inicializado como vazio
        x_unique = list(np.unique(X)) # Encontra todos os valores únicos em X e guarda-os numa lista
        for i in range(1, len(x_unique)):
            # Calcula a média entre cada 2 valores da lista - representa a divisão na árvore
            average = (x_unique[i - 1] + x_unique[i]) / 2.0
            split_values.add(average) # Adiciona o split value à lista

        return list(split_values)

    def _find_best_split(self, X, target, n_features, class_weights):

        '''Encontra o melhor atributo e o melhor valor para fazer o split'''

        # Sample random subset of features
        subset = random.sample(list(range(0, X.shape[1])), n_features) # Atributos do dataset
        max_gain, max_col, max_val = None, None, None


        # Para cada atributo, calcula os split values e escolhe o melhor com base no critério selecionado
        for column in subset:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                if self.loss is None:
                    # Random forest
                    splits = split(X[:, column], target["y"], value)
                    gain = self.criterion(target["y"], splits, class_weights)
                else:
                    # Gradient boosting
                    left, right = split_dataset(X, target, column, value, return_X=False)
                    gain = xgb_criterion(target, left, right, self.loss)

                if (max_gain is None) or (gain > max_gain):
                    max_col, max_val, max_gain = column, value, gain
        return max_col, max_val, max_gain # Retorna o melhor atributo e o melhor valor para o split

    def _train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01, class_weights = None):
        '''Treina a árvore de decisão'''
        try:
            assert X.shape[0] > min_samples_split 
            assert max_depth > 0

            if max_features is None:
                max_features = X.shape[1]

            column, value, gain = self._find_best_split(X, target, max_features, class_weights)
            assert gain is not None
            if self.regression:
                assert gain != 0
            else:
                assert gain > minimum_gain

            self.column_index = column
            self.threshold = value
            self.impurity = gain

            # Split dataset
            left_X, right_X, left_target, right_target = split_dataset(X, target, column, value)

            # Grow left and right child
            self.left_child = WeightedTree(self.regression, self.criterion, self.n_classes)
            self.left_child._train(
                left_X, left_target, max_features, min_samples_split, max_depth - 1, minimum_gain, class_weights
            )

            self.right_child = WeightedTree(self.regression, self.criterion, self.n_classes)
            self.right_child._train(
                right_X, right_target, max_features, min_samples_split, max_depth - 1, minimum_gain, class_weights
            )
        except AssertionError:
            self._calculate_leaf_value(target, class_weights) 

    def train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01, loss=None):
      
        if not isinstance(target, dict):
            target = {"y": target}

        # Loss for gradient boosting
        if loss is not None:
            self.loss = loss

        if not self.regression:
            self.n_classes = len(np.unique(target['y']))
    
        class_weights = calculate_class_weights(target['y'])
    
        self._train(X, target, max_features=max_features, min_samples_split=min_samples_split,
                    max_depth=max_depth, minimum_gain=minimum_gain, class_weights=class_weights)
        


    def _calculate_leaf_value(self, targets, class_weights):
        '''Calcula o valor da folha (classe)'''
        if self.loss is not None:
            # Gradient boosting
            self.outcome = self.loss.approximate(targets["actual"], targets["y_pred"])
        else:
            # Random Forest
            if self.regression:
                # Mean value for regression task
                self.outcome = np.mean(targets["y"])
            else:
                # Probability for classification task
                weighted_counts = [np.sum(targets["y"] == i) * class_weights.get(i, 1) for i in range(self.n_classes)]
                self.outcome = np.array(weighted_counts) / sum(weighted_counts)
                # self.outcome = np.bincount(targets["y"], minlength=self.n_classes) / targets["y"].shape[0]

    def predict_row(self, row):
        """Prevê a classe de uma linha"""
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result
    

class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.

        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()
    


class WeightedRandomForest(BaseEstimator):
    def __init__(self, n_estimators=200, max_features=None, min_samples_split=10, max_depth=None, criterion=None):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features
        self._train()

    def _train(self):
        for tree in self.trees:
            tree.train(
                self.X,
                self.y,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )

    def _predict(self, X=None):
        raise NotImplementedError()
    
    

class WeightedRandomForestClassifier(WeightedRandomForest):
    def __init__(self, n_estimators=1000, max_features=None, min_samples_split=10, max_depth=1000, criterion="entropy"):
        super(WeightedRandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            criterion=criterion,
        )


        if criterion == "entropy":
            self.criterion = information_gain
        else:
            raise ValueError()

        # Initialize empty trees
        for _ in range(self.n_estimators):
            self.trees.append(WeightedTree(criterion=self.criterion))
       
    def _predict(self, X=None):
        y_shape = np.unique(self.y).shape[0]
        predictions = np.zeros((X.shape[0], y_shape))

        for i in range(X.shape[0]):
            row_pred = np.zeros(y_shape)
            for tree in self.trees:
                row_pred += tree.predict_row(X[i, :])

            row_pred /= self.n_estimators
            predictions[i, :] = row_pred

        return np.argmax(predictions, axis=1)
    

# #Performance Report
# def generateClassificationReport(y_test,y_pred):
#     print(classification_report(y_test,y_pred))
#     print(confusion_matrix(y_test,y_pred))    
#     print('Accuracy: ', accuracy_score(y_test,y_pred))
#     print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_pred))
#     print('F1 score: ', f1_score(y_test, y_pred))
#     print('Roc_auc_score: ', roc_auc_score(y_test, y_pred))
#     print('Recall: ', recall_score(y_test, y_pred))