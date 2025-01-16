import numpy as np
from scipy import stats
import random

def f_entropy(p):
    # Convert values to probability
    p = np.bincount(p) / float(p.shape[0])

    ep = stats.entropy(p)
    if ep == -float("inf"):
        return 0.0
    return ep


def information_gain(y, splits):
    splits_entropy = sum([f_entropy(split) * (float(split.shape[0]) / y.shape[0]) for split in splits])
    return f_entropy(y) - splits_entropy


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

class Tree(object):
    """Recursive implementation of decision tree."""

    def __init__(self, regression=False, criterion=None, n_classes=None):
        self.regression = regression
        self.impurity = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.criterion = criterion
        self.loss = None
        self.n_classes = n_classes  # Only for classification
        self.left_child = None
        self.right_child = None

    @property
    def is_terminal(self):
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X):
        """Find all possible split values."""
        split_values = set()

        # Get unique values in a sorted order
        x_unique = list(np.unique(X))
        for i in range(1, len(x_unique)):
            # Find a point between two values
            average = (x_unique[i - 1] + x_unique[i]) / 2.0
            split_values.add(average)

        return list(split_values)

    def _find_best_split(self, X, target, n_features):
        """Find best feature and value for a split. Greedy algorithm."""

        # Sample random subset of features
        subset = random.sample(list(range(0, X.shape[1])), n_features)
        max_gain, max_col, max_val = None, None, None

        for column in subset:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                if self.loss is None:
                    # Random forest
                    splits = split(X[:, column], target["y"], value)
                    gain = self.criterion(target["y"], splits)
                else:
                    # Gradient boosting
                    left, right = split_dataset(X, target, column, value, return_X=False)
                    gain = xgb_criterion(target, left, right, self.loss)

                if (max_gain is None) or (gain > max_gain):
                    max_col, max_val, max_gain = column, value, gain
        return max_col, max_val, max_gain

    def _train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01):
        try:
            # Exit from recursion using assert syntax
            assert X.shape[0] > min_samples_split
            assert max_depth > 0

            if max_features is None:
                max_features = X.shape[1]

            column, value, gain = self._find_best_split(X, target, max_features)
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
            self.left_child = Tree(self.regression, self.criterion, self.n_classes)
            self.left_child._train(
                left_X, left_target, max_features, min_samples_split, max_depth - 1, minimum_gain
            )

            self.right_child = Tree(self.regression, self.criterion, self.n_classes)
            self.right_child._train(
                right_X, right_target, max_features, min_samples_split, max_depth - 1, minimum_gain
            )
        except AssertionError:
            self._calculate_leaf_value(target)

    def train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01, loss=None):
        """Build a decision tree from training set.

        Parameters
        ----------

        X : array-like
            Feature dataset.
        target : dictionary or array-like
            Target values.
        max_features : int or None
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the tree.
        minimum_gain : float, default 0.01
            Minimum gain required for splitting.
        loss : function, default None
            Loss function for gradient boosting.
        """

        if not isinstance(target, dict):
            target = {"y": target}

        # Loss for gradient boosting
        if loss is not None:
            self.loss = loss

        if not self.regression:
            self.n_classes = len(np.unique(target['y']))

        self._train(X, target, max_features=max_features, min_samples_split=min_samples_split,
                    max_depth=max_depth, minimum_gain=minimum_gain)


    def _calculate_leaf_value(self, targets):
        """Find optimal value for leaf."""
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
                self.outcome = np.bincount(targets["y"], minlength=self.n_classes) / targets["y"].shape[0]

    def predict_row(self, row):
        """Predict single row."""
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

class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=1000, max_features=None, min_samples_split=10, max_depth=None, criterion=None):
        """Base class for RandomForest.

        Parameters
        ----------
        n_estimators : int
            The number of decision tree.
        max_features : int
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the tree.
        criterion : str
            The function to measure the quality of a split.
        """
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


class RandomForestClassifierM(RandomForest):
    def __init__(self, n_estimators=200, max_features=None, min_samples_split=10, max_depth=None, criterion="entropy"):
        super(RandomForestClassifierM, self).__init__(
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
            self.trees.append(Tree(criterion=self.criterion))

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


# class RandomForestRegressor(RandomForest):
#     def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion="mse"):
#         super(RandomForestRegressor, self).__init__(
#             n_estimators=n_estimators,
#             max_features=max_features,
#             min_samples_split=min_samples_split,
#             max_depth=max_depth,
#         )

#         if criterion == "mse":
#             self.criterion = mse_criterion
#         else:
#             raise ValueError()

#         # Initialize empty regression trees
#         for _ in range(self.n_estimators):
#             self.trees.append(Tree(regression=True, criterion=self.criterion))

#     def _predict(self, X=None):
#         predictions = np.zeros((X.shape[0], self.n_estimators))
#         for i, tree in enumerate(self.trees):
#             predictions[:, i] = tree.predict(X)
        # return predictions.mean(axis=1)