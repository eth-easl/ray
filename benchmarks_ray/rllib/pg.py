from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)

# Initialize Ray.
import ray

import tensorflow as tf

ray.init(address="auto")

pg = placement_group(["GPU": 1, "CPU": 1], strategy="STRICT_PACK")
# Wait until placement group is created.
ray.get(pg.ready())

print(placement_group_table(pg))


@ray.remote(num_gpus=1)
class GPUActor:
  def __init__(self):
    print(tf.config.list_physical_devices('GPU'))

# Create GPU actors on a gpu bundle.
gpu_actors = [GPUActor.options(
    placement_group=pg,
    # This is the index from the original list.
    # This index is set to -1 by default, which means any available bundle.
    placement_group_bundle_index=0) # Index of gpu_bundle is 0.
.remote() for _ in range(2)]


ray.get(gpu_actors)