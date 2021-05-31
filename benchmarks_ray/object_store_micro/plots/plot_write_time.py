import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('write_objects_workers.csv', header=0)
print(df)


columns = ['Workers', 'Total(ms)', 'Time_put(ms)']
lines = len(columns)

workers = np.asarray(df['Workers'])
time=[]
for i in range(1,lines):
    time.append(np.asarray(df[columns[i]]))

fig, ax = plt.subplots(figsize=(10,8))

labs=[]
lns=[]

#for i in range(lines):
line=plt.bar(workers, time[1], width=0.5, color='#6187a5')
#lns+=line

#labs = [l.get_label() for l in lns]

#ax.set_xscale('log')

#ax.set_xticks(obj_size)
#ax.set_xticklabels(x_labels)

ax.set_xlabel('number of workers', fontsize=14)
ax.set_ylabel('time(ms)', fontsize=14)
ax.set_title('ray.put() 200 MB', fontsize=14)
#ax.legend(lns, labs)


fig.savefig('write_workers_200MB.png', bbox_inches = 'tight')

