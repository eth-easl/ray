import numpy as np
import pandas as pd

import ray
import ray.autoscaler.sdk

from time import sleep, perf_counter
from tqdm import tqdm
import time
import pandas as pd

ARRAY_SIZE=2*2**30
#num_workers=[1]
num_workers=[1,3,6,12,15,18,24]
runs=10

a = np.ones(ARRAY_SIZE, dtype=np.uint8)

@ray.remote(num_cpus=1)
def sum_arr(f):
    return np.sum(f)


ray.init(address="auto")
ref = ray.put(a)

measurements=[]

for w in num_workers:
    start_time = time.time()
    obj_ref = [sum_arr.remote(ref) for i in range(w)]
    results = ray.get(obj_ref)
    end_time = time.time()
    duration = end_time-start_time
    print("workers: ", w, "time: ", duration)
    for r in results:
        assert (r==ARRAY_SIZE)
    
    measurements.append([w, duration])

df = pd.DataFrame(np.asarray(measurements), columns=['workers', 'time'])
df.to_csv('/home/ubuntu/object_store_micro/results/read_objects.csv')
