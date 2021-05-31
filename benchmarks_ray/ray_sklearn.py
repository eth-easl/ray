import numpy as np
from joblib import parallel_backend # added line.
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
import ray

import time
from datetime import datetime as dt


from ray.util.joblib import register_ray # added line.
register_ray() # added line.

param_space = {
    'C': np.logspace(-6, 6, 2),
    'gamma': np.logspace(-8, 8, 2),
    'tol': np.logspace(-4, -1, 2),
    'class_weight': [None, 'balanced'],
}

model = SVC(kernel='rbf')
search = GridSearchCV(model, param_space, cv=5, verbose=2)
c = load_digits()

ray.init(address="auto")

# ar = np.zeros(100000, dtype=float)
# ref = ray.put(ar)
# print(ref)

start = dt.now()
with parallel_backend('ray', n_jobs=2): # Ray manages the runtime  of the parallel jobs of gridsearch?
    search.fit(c.data, c.target)
    #sleep(10)
end = dt.now()
print("Elapsed Time: {}".format((end-start).total_seconds()))
#print("results: ", search.cv_results_)