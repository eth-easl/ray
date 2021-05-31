import ray
from ray.util.dask import ray_dask_get
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from time import sleep

# Start Ray.
# Tip: If you're connecting to an existing cluster, use ray.init(address="auto").
ray.init(address="auto")
#ray.init()
#d_arr = da.from_array(np.random.randint(0, 1000, size=(256, 256)))

# The Dask scheduler submits the underlying task graph to Ray.
#d_arr.mean().compute(scheduler=ray_dask_get)

# Set the scheduler to ray_dask_get in your config so you don't have to specify it on
# each compute call.
#dask.config.set()
cols = ['col_' + str(i) for i in range(2**12)]

df = dd.from_pandas(pd.DataFrame(
    np.random.randint(0, 100, size=(2**12, 2**12)),
    columns=cols), npartitions=16)

means_group = []
means = []
devs = []

print('nans')
has_nan = df.isnull().values.any().compute(scheduler=ray_dask_get, num_workers=16)

print('means')
means = df.mean(axis=0).compute(scheduler=ray_dask_get, num_workers=16)

print('devs')
devs = df.std(axis=0).compute(scheduler=ray_dask_get, num_workers=16)

print('groupby')
for i in range(500):
    # find mean, dev etc when groupping at each column
    print(i)
    means_group.append(df.groupby(cols[i]).mean().compute(scheduler=ray_dask_get, num_workers=16))