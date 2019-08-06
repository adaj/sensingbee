import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold


def kfold(cv, scoring, X, y):
    scores = []
    for train_index, test_index in KFold(n_splits=cv).split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scores.append(scoring(y_test, X_test))
    return np.array(scores)


class Model(object):

    def __init__(self, base_estimator,  tuning_conf=False):
        self.base_estimator = base_estimator
        self.tuning_conf = tuning_conf
        if self.tuning_conf:
            try:
                self.base_estimator = RandomizedSearchCV(base_estimator, n_iter=self.tuning_conf['iterations'],
                                        param_distributions=self.tuning_conf['params'],
                                        cv=self.tuning_conf['cv'], scoring=self.tuning_conf['scoring'],
                                        refit='r2')
            except:
                self.base_estimator = GridSearchCV(base_estimator,
                                        param_grid=self.tuning_conf['params'],
                                        cv=self.tuning_conf['cv'], scoring=self.tuning_conf['scoring'],
                                        refit='r2')
        else:
            self.base_estimator = base_estimator

    def fit(self, X, y):
        self.base_estimator.fit(X, np.ravel(y))
        try:
            self.feature_importances_ = pd.DataFrame(self.base_estimator.best_estimator_.feature_importances_,
                      index=X.columns, columns=['Importance'])
        except:
            pass
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)
