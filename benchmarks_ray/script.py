import os

os.system("rllib train --ray-address 10.138.0.51:6379 -f cartpole-appo.yaml -vv")
#os.system("rllib train -f rllib/pong-apex.yaml")

#os.system("python cifar_tf_example.py --address='10.138.0.51:6379'")
