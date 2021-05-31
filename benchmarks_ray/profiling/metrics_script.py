import ray
import json

ray.init(address='auto')
from pprint import pprint
res = ray.nodes()

with open('metrics_config,json', 'w') as outfile:
    json.dump(res, outfile)
