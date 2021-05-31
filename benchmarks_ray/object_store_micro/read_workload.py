import numpy as np
import pandas as pd

import ray
import ray.autoscaler.sdk
import ray.services

from time import sleep, perf_counter
from tqdm import tqdm
import time
import pandas as pd

runs=10
num_objects=10
run_time = 50

ray.init(address="auto")

obj_list = []
ref_list = []
res = ray.cluster_resources()
res_keys = res.keys()
nodes = []

for e in res_keys:
    if ('node' in e):
        nodes.append(e)

local_hostname = ray.services.get_node_ip_address()
driver_node_id = f"node:{local_hostname}"

# # Check to make sure the node id resource exists
assert driver_node_id in nodes
nodes.remove(driver_node_id)

worker_node_id_1 = nodes[0]
#worker_node_id_2 = nodes[1]

print(driver_node_id, worker_node_id_1)

# Obtain the local ip address
local_hostname = ray.services.get_node_ip_address()
assert isinstance(local_hostname, str)

 # Ray has a predefined "node id" resource for locality placement
node_id = f"node:{local_hostname}"
print(node_id)
# Check to make sure the node id resource exists
assert node_id in ray.cluster_resources()

measurements=[]

a = 'a'* 10**3
#a = np.ones(obj_size, dtype=np.uint8)

@ray.remote(num_cpus=8)
class worker_1():
    def put_obj(self, sz):

        ref_list = []
        #print("hello!!")

        for i in range(num_objects):
            ref = ray.put(a)
            ref_list.append(ref)

        #self.ref_list = ref_list

        #print('done!', self.ref_list)
        return ref_list


def benchmark(obj_size, w):


    @ray.remote
    def get_arr():
        #sleep_time = sleep_us/1000000
        op=0
        start = time.time()
        #print("hello!")
        #print(len(ref_list))
        #while (time.time() - start < run_time):
        while (op < num_objects):
            ar = ray.get(ref_list[op])
            op += 1
            #sleep(sleep_time)
            #del ar
        end = time.time()
        #print(end-start)
        return (op, (end - start))

    # Create a remote function with the given resource label attached
    local_get_func = get_arr.options(resources={worker_node_id_1: 0.01})

    w1 = worker_1.remote()
    ref = [w.put_obj.remote(obj_size) for w in [w1]]
    [ref_list] = ray.get(ref)

    sleep(10)

    obj = local_get_func.remote()
    ray.get(obj)

    sleep(10)
    num_workers = [w]

    for i in num_workers:
        #for s in sleep_us_list:
            print("--------------------- workers: ", w)
            obj_ref = [local_get_func.remote() for j in range(i)]

            start = time.time()
            ready_refs, remaining_refs = ray.wait(obj_ref, num_returns=w, timeout=run_time)
            results = ray.get(ready_refs)
            end=time.time()

            total_time = min(run_time, end-start)


            print("returned: ", len(results))

            if (len(results) > 0):

                op_per_worker = [res[0] for res in results]
                time_per_worker = [res[1] for res in results]

                total_ops = np.sum(np.asarray(op_per_worker))
                avg_ops = np.median(np.asarray(op_per_worker))
                max_time = max(time_per_worker)

                iops_per_worker = [(res[0]/res[1]) for res in results]
                average_iops = np.median(np.asarray(iops_per_worker))
                min_iops = min(iops_per_worker)
                max_iops = max(iops_per_worker)

                #put_duration = np.median(np.asarray(results[0]))
                sz_gb = obj_size/10**9
                th_per_worker = [sz_gb*i for i in iops_per_worker]
                average_th = np.median(np.asarray(th_per_worker))


                system_iops =  total_ops/total_time
                system_th =  system_iops * sz_gb

                average_latency = 1000.0/average_iops

                for r in remaining_refs:
                    ray.cancel(r, force=True)

                if (w>0):
                    print(max(op_per_worker), min(op_per_worker))
                    print("object size: ", obj_size, "total ops: ", total_ops, "avg throughput (GB/s): ", average_th, "avg iops: ", average_iops, "avg latency(ms): ", average_latency)
                    print("system iops: ", system_iops, "system throughput: ", system_th)
                    print("max time: ", max_time, "total time: ", total_time)
                    print("average latency(ms): ", average_latency, "max latency: ", 1000.0/min_iops, "min latency: ", 1000.0/max_iops)

                return [w, obj_size, system_iops, system_th, average_iops, average_th, average_latency, avg_ops, len(results), total_time]

            else:
                return [w,obj_size,0,0,0,0,0,0,0,run_time]



# warm-up

obj_sizes=[10**3]
#sleep_us_list = [0, 0, 100, 200, 300, 400, 500, 1000]
num_workers = [1]
#num_workers=[1]
for o in obj_sizes:
    for w in num_workers:
	benchmark(o,w)        
	#res = []
        #for r in range(5):
            #res.append(benchmark(o, w))
            #sleep(30)
        #res = res[1:]
        #print(res)
        #res_np = np.asarray(res)
        #measurements.append(np.median(res_np, axis=0))


#df = pd.DataFrame(np.asarray(measurements), columns=['workers', 'obj_size','System IOPS', 'System Throughput(GB/s)', 'Worker IOPS', 'Worker Throughput(GB/s)', 'Avg Latency(ms)', 'Avg Ops', 'Returned', 'Total_time'])
#print(df)
#df.to_csv('/home/ubuntu/object_store_micro/results/read_objects_throughput_local_new2.csv')

