import numpy as np

import ray
import ray.autoscaler.sdk

from time import sleep, perf_counter
from tqdm import tqdm
import time
import pandas as pd
import ray.services

ARRAY_SIZE = 2*10**8
num=6
num_workers=64
runs = 5000
run_time = 5

#a = np.ones(ARRAY_SIZE, dtype=np.uint8)
df= pd.DataFrame()
measurements =[]

@ray.remote(num_cpus=1)
def put_objects():
    #refs=[]
    for i in range(1):
        a = np.ones(10**i, dtype=np.uint8)
        start = time.time()
        for j in range(runs):
            ref = ray.put(a)
        end = time.time()
        duration = (end-start)/runs
        #duration = end-start
        size_gb = 10**i/10**9
        #print(10**i, " bytes, ", duration, " sec ", size_gb/duration  ," GB/s")

        #print(10**i, " bytes, ", duration, " sec ", runs/duration  ," IOPS")
        measurements.append([10**i, runs/duration])
        #measurements.append(duration)

    print(measurements)
    df = pd.DataFrame(np.asarray(measurements))
    print(df)
    #df.to_csv("/home/ubuntu/object_store_micro/write_objects_iops.csv", header=None)
    return


@ray.remote
def put_objects_2(sz):
    a = np.ones(sz, dtype=np.uint8)
    op = 0
    start = time.time()
    #while (time.time() - start < run_time):
    while (op<1):
        #for j in range(runs):
        ref = ray.put(a)
        # print(ref)
        op+=1
    end = time.time()
    #print(op)
    return (op, (end - start)*1000)

ray.init(address="auto")

res = ray.cluster_resources()
res_keys = res.keys()
nodes = []

for e in res_keys:
    if ('node' in e):
        nodes.append(e)

local_hostname = ray.services.get_node_ip_address()
print(local_hostname)

driver_node_id = f"node:{local_hostname}"

local_put_func_2 = put_objects_2.options(resources={driver_node_id: 0.01})

# fill obj store
a  = 'a' * 2**20

refs=[]
for i in range(1000):
    refs.append(ray.put(a))
sleep(30)

ray.get(refs)
sleep(30)

#obj_sizes=[10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
obj_sizes=[10**8]
workers = [1]
for w in workers:
    print("--------------------- workers: ", w)
    for sz in obj_sizes:
        results = []
        start = time.time()
        for j in range(1):
            obj_ref = [local_put_func_2.remote(sz) for i in range(w)]
            results = ray.get(obj_ref)
        end = time.time()

        op_per_worker = np.asarray([res[0] for res in results])
        time_per_worker = np.asarray([res[1] for res in results])
        latency_per_worker = np.asarray([time_per_worker[i]/op_per_worker[i] for i in range(w)])

        avg_ops = np.median(op_per_worker)
        avg_time = np.median(time_per_worker)
        avg_lat = np.median(latency_per_worker)
        # total_ops = np.sum(np.asarray(op_per_worker))
        # max_time = max(time_per_worker)

        # iops_per_worker = [(res[0]/res[1]) for res in results]
        # average_iops = np.median(np.asarray(iops_per_worker))

        # #put_duration = np.median(np.asarray(results[0]))
        # sz_gb = sz/10**9
        # th_per_worker = [sz_gb*i for i in iops_per_worker]
        # average_th = np.median(np.asarray(th_per_worker))

        # system_iops =  total_ops/max_time
        # system_th =  system_iops * sz_gb

        # print("object size: ", sz, "total ops: ", total_ops, "avg throughput (GB/s): ", average_th, "avg iops: ", average_iops)
        # print("system iops: ", system_iops, "system throughput: ", system_th)
        # print("max time: ", max_time)

        print("Objects: ", avg_ops, " Time: ", avg_time, " Latency: ", avg_lat)

        measurements.append([w, sz, avg_lat])
        sleep(30)

df = pd.DataFrame(data=np.asarray(measurements), columns=['workers', 'obj_size','Time_put(ms)'])
print(df)
#df.to_csv("/home/ubuntu/object_store_micro/results/spilling/write_obj_1_worker.csv")

