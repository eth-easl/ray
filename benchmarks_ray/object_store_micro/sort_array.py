import numpy as np
import pandas as pd

import ray
import ray.autoscaler.sdk

from time import sleep, perf_counter
from tqdm import tqdm
import time

ARRAY_SIZE=2**30
num_workers=3

a = np.random.randint(255, size=ARRAY_SIZE, dtype=np.uint8)
#sort_a = np.sort(a)

@ray.remote(num_cpus=8)
def sort_arr(arr):
    return np.sort(arr)

def merge(l):
    # merge the sorted arrays here
    return

ray.init(address="auto")
ref = ray.put(a)

n = ARRAY_SIZE/num_workers

start_time = time.time()

obj_ref = [sort_arr.remote(ref) for i in range(num_workers)]
results = ray.get(obj_ref)

end_time = time.time()

#for ar in results:
#    assert (ar == sort_a).all()

#print(results)
print("Duration: ", end_time-start_time)
