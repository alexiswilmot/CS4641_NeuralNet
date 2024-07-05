import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import ExtraTreeClassifier


class RandomForest(object):
    def __init__(self, n_estimators, max_depth, max_features, random_seed=None):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_seed = random_seed
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [
            ExtraTreeClassifier(max_depth=max_depth, criterion="entropy")
            for i in range(n_estimators)
        ]

    def _bootstrapping(self, num_training, num_features, random_seed=None):
        """
        TODO:
        - Randomly select a sample dataset of size num_training with replacement from the original dataset.
        - Randomly select certain number of features (num_features denotes the total number of features in X,
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.

        Args:
        - num_training: number of data points in the bootstrapped dataset.
        - num_features: number of features in the original dataset.

        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        Hint 1: Please use np.random.choice. First get the row_idx first, and then second get the col_idx.
        Hint 2:  If you are getting a Test Failed: 'bool' object has no attribute 'any' error, please try flooring, or converting to an int, the number of columns needed for col_idx. Using np.ceil() can cause an autograder error.
        """
        if random_seed != None:
            np.random.seed(random_seed)
        # randomly select a sample dataset of size num_training with raplacement rom original dataset
        inds = np.random.choice(num_training, size = num_training, replace=True)
        feet = int(np.floor(self.max_features * num_features))
        # replace false
        col = np.random.choice(num_features, size=feet, replace=False)
        return inds, col
        # data = 
        raise NotImplementedError()

    def bootstrapping(self, num_training, num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        np.random.seed(self.random_seed)
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        X: NxD numpy array, where N is number
           of instances and D is the dimensionality of each
           instance
        y: 1D numpy array of size (N,), the predicted labels
        Returns:
            None. Calling this function should train the decision trees held in self.decision_trees
        """
        # 
        instances = X.shape[0]
        dim = X.shape[1]
        self.feature_indices = []
        self.bootstraps_row_indices = []
        # for each tree, initialize bootstrap datasets
        self.bootstrapping(instances, dim)
        for i in range(self.n_estimators):
            # bootstrapping - get random ones
            rows = self.bootstraps_row_indices[i]
            cols = self.feature_indices[i]
            # row, col = self._bootstrapping(instances, dim, self.random_seed)
            # self.feature_indices.append(col)
            # self.bootstraps_row_indices.append(row)
            # features x and labels y
            xBoot = X[rows][:, cols]
            yBoot = y[rows]
            # train decision trees again
            self.decision_trees[i].fit(xBoot, yBoot)
        return
        # boot_row = self.bootstraps_row_indices
        # boot_feat = self.feature_indices
        #raise NotImplementedError()

    def OOB_score(self, X, y):
        # helper function. You don't have to modify it
        # This function computes the accuracy of the random forest model predicting y given x.
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(
                        self.decision_trees[t].predict(
                            np.reshape(X[i][self.feature_indices[t]], (1, -1))
                        )[0]
                    )
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

    def predict(self, X):
        N = X.shape[0]
        y = np.zeros((N, 7))
        for t in range(self.n_estimators):
            X_curr = X[:, self.feature_indices[t]]
            y += self.decision_trees[t].predict_proba(X_curr)
        pred = np.argmax(y, axis=1)
        return pred

    def plot_feature_importance(self, data_train):
        """
        TODO:
        -Display a bar plot showing the feature importance of every feature in
        the first decision tree from the tuned random_forest
        -Make sure to sort thebars in descending order and remove any features with feature
        importance of 0
        -Gini importance
        Args:
            data_train: This is the orginal data train Dataframe containg data AND labels.
                Hint: you can access labels with data_train.columns
        Returns:
            None. Calling this function should simply display the aforementioned feature importance bar chart
        """
        # make plot for first tree only
        
        # 
        raise NotImplementedError()

    def select_hyperparameters(self):
        """
        Hyperparameter tuning Question
        TODO: assign a value to n_estimators, max_depth, max_features
        Args:
            None
        Returns:
            n_estimators: int number (e.g 2)
            max_depth: int number (e.g 4)
            max_features: a float between 0.0-1.0 (e.g 0.1)
        """
        #return 15, 15, .7
        return 1, 14, 1
        raise NotImplementedError()
