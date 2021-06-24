import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

colors = ['#6187a5', '#896689', '#f4a680', '#cec591']

df = pd.read_csv('rllib_fails.csv')
print(df)
num_workloads = 4

fails = list(df['Fails'])
lat = list(df['Runtime(s)'])
slowdown=[x/lat[0] for x in lat]

'''
lat = list(df['Bandwidth(Gbps)'])
perc = ['10', '35', '50', '75', '100']

ips = []
for p in perc:
    t = 'ips_' + p
    ips.append(list(df[t]))

base_ips = 336.7
base_times = 76.1

print(ips)
data_ips=[]
for i in range(len(ips[0])):
    l=[]
    for lst in ips:
        s = (base_ips-lst[i])/base_ips
        l.append(s*100)
    data_ips.append(l)

times = []
for p in perc:
    t = 'time_' + p
    times.append(list(df[t]))
data_times=[]
for i in range(len(times[0])):
    l=[]
    for l in times:
        l.append(l[i])
    data_times.append(l)
# slowdown = [list(df['resnet_gpu_slowdown'])[1:]]
# ips = [list(df['resnet_gpu_ips'])]


print(data_ips, data_times)
labels = ['10 Gbps', '16 Gbps', '40 Gbps', '100 Gbps']
'''

fig, ax = plt.subplots(figsize=(12,10))
x = np.arange(num_workloads)
#x_labels = ['10%', '35%', '50%', '75%', '100%']
x_labels = ['0', '1', '2', '4']
barWidth = 0.35

#for i in range(1):
line=plt.bar(x, slowdown, width=barWidth, color=colors[1], alpha=0.9, linewidth=10)

plt.xlabel('Num of failures', fontsize = 15)
plt.ylabel('Runtime slowdown', fontsize = 15)
plt.xticks(x, x_labels, fontsize=14)
plt.title("RL slowdown when killing workers (1 fault every 2 sec.)", fontsize=14)

#plt.legend(loc=2, fontsize=12)
#plt.grid(axis='y',  linestyle='--', linewidth=0.5)
#plt.show()
plt.savefig("ft_rllib.png",  bbox_inches = 'tight')

# num_workloads = 5
# x =  np.arange(num_workloads)
# rem_perc = [30, 48, 30, 10, 18]
# bytes_transf = [200, 30, 60, 0.02, 36]
# cpu_util = [50, 49.6, 43, 80, 60]
# mem_util = [25.3, 14.3, 15, 15, 13]
# labels = ['train', 'serve', 'rl appo', 'rl alphazero', 'rl dqn']
# fig = plt.figure(figsize = (7, 5))

# plt.bar(x, mem_util, color =colors[3], width=0.4)

# plt.xlabel("Workload", fontsize=11)
# #plt.ylabel("Percentage(%) of remote accesses", fontsize=11)
# # plt.ylabel("GB transferred between nodes", fontsize=11)
# # plt.yscale('log')
# # plt.yticks([0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])

# plt.ylabel("Mem utilization(%)", fontsize=11)


# #plt.title("")
# plt.xticks([r for r in range(num_workloads)], labels, fontsize=11)
# plt.savefig("mem_util.png",  bbox_inches = 'tight')