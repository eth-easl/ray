import os
import psutil
import time
from random import randrange

def findProcessIdByName(processName):
    '''
    Get a list of all the PIDs of a all the running process whose name contains
    the given string processName
    '''
    listOfProcessObjects = []
    #Iterate over the all the running process
    for proc in psutil.process_iter():
       try:
           pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
           # Check if process name contains the given name string.
           if processName.lower() in pinfo['name'].lower() :
               listOfProcessObjects.append(pinfo)
       except (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess) :
           pass
    return listOfProcessObjects


# do it at a regular time interval
interval_sec=2
num_fails=0
max_fails=5

while(num_fails<max_fails):
    listOfProcessIds = findProcessIdByName('RolloutWorker')
    print(listOfProcessIds)
    if len(listOfProcessIds) > 0:
        idx=randrange(10)
        elem=listOfProcessIds[idx]
        processID = elem['pid']
        print(processID)
        cmd = 'kill -9 ' + str(processID)
        os.system(cmd)
        print(cmd)
    time.sleep(2)
    num_fails+=1

