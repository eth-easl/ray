import ray
import numpy as np
import time

@ray.remote(max_restarts=5, max_task_retries=-1)
class Foo():
    def __init__(self):
        self.a=np.ones(10**8, dtype=np.uint8)
    
    def op(self):
        for i in range(200):
            l=np.sum(self.a)

        return self.a

ray.init(address="auto")

start = time.time()
actors=[Foo.remote() for i in range(1)]

refs=[i.op.remote() for i in actors]
ars=ray.get(refs)
end = time.time()

for ar in ars:
    print(ar.shape)

print ("It took: ", end-start, " sec")
