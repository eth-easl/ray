# Keep this here for https://github.com/ray-project/ray/issues/11547
from sklearn.model_selection import GridSearchCV
# Replace above line with:
from ray.tune.sklearn import TuneGridSearchCV

import ray

# Other imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
import numpy as np

# Create dataset
X, y = make_classification(
    n_samples=11000,
    n_features=1000,
    n_informative=50,
    n_redundant=0,
    n_classes=10,
    class_sep=2.5)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1000)

# Example parameters to tune from SGDClassifier
parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}


tune_search = TuneGridSearchCV()

import time  # Just to compare fit times
start = time.time()
tune_search.fit(x_train, y_train)
end = time.time()
print("Tune GridSearch Fit Time:", end - start)
# Tune GridSearch Fit Time: 15.436315774917603 (for an 8 core laptop)
