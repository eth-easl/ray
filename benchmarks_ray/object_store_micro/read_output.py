import numpy as np


f = open('output.txt', 'r')
write=False
read=True
'''
if write:
    cpy=[]
    ser=[]
    plasma_seal=[]
    plasma_create=[]
    for line in f.readlines():
        tokens=line.split(' ')
        print(tokens)
        if 'Memcpy' in tokens:
            if (len(tokens) > 6):
                cpy.append(float(tokens[5]))
        elif 'serialization' in tokens:
            if (len(tokens) > 6):
                ser.append(float(tokens[5]))
        elif 'Create' in tokens:
            if (len(tokens) > 8):
                plasma_create.append(float(tokens[7]))
        elif 'Communication' in tokens:
            if (len(tokens) > 7):
                plasma_seal.append(float(tokens[6])/1000)
    cpy_med = np.median(np.asarray(cpy))
    ser_med = np.median(np.asarray(ser))
    plasma_c_med = np.median(np.asarray(plasma_create))
    plasma_s_med = np.median(np.asarray(plasma_seal))

    print("Memcpy: ", cpy_med, " Ser: ", ser_med, " Plasma Create: ", plasma_c_med, "Plasma Seal: ", plasma_s_med)

if read:
    raylet=[]
    deser=[]
    plasma=[]
    total=[]
    #for line in f.readlines():
    while True:
        line = f.readline()
        if not line:
            break
        tokens=line.split(' ')
        if ('2nd' in tokens):
           for i in range(8):
                l = f.readline()
                tokens = l.split()
                print(tokens)
                if 'deserialization' in tokens:
                    if (len(tokens) >= 6):
                        deser.append(float(tokens[-2]))
                elif 'Raylet' in tokens and 'Total' not in tokens:
                    if (len(tokens) >= 8):
                        raylet.append(float(tokens[-2])/1000)
                elif 'Getting' in tokens and 'Plasma' in tokens:
                    if (len(tokens) >= 9):
                        plasma.append(float(tokens[-2])/1000)

    deser_med = np.median(np.asarray(deser))
    ray_med = np.median(np.asarray(raylet))
    plasma_med = np.median(np.asarray(plasma))
'''


#read from raylet
#transfer=[]
get=[]
put=[]
pull=[]
push=[]
rest=[]
prep = []
send = []
r1 = open('raylet1.out', 'r')
r2 = open('raylet2.out', 'r')

sz_gb=0.001

for line in r2.readlines():
    tokens=line.split(' ')
    #if 'Writing' in tokens:
    if ('Writing' in tokens):
        num=tokens[-1][:-4] #sec
        print(tokens)
        rest.append(float(num)*1000) # make it ms
    #if ('Writing' in tokens):
    if ('ReceiveObjectChunk' in tokens and 'took' in tokens):
        print(tokens)
        num=tokens[-1][:-4] #sec
        print(tokens)
        put.append(float(num)*1000) # make it ms
    if 'Getting' in tokens:
        num=tokens[-1][:-4] #sec
        print(tokens)
        pull.append(float(num)*1000) # make it ms

for line in r1.readlines():
    if 'Reading the object' in line:
        tokens=line.split(' ')
        num=tokens[-1][:-4] #sec
        print(tokens)
        get.append(float(num)*1000) # make it ms

    if 'Pushing the object' in line:
        tokens=line.split(' ')
        num=tokens[-1][:-4] #sec
        print(tokens)
        push.append(float(num)*1000) # make it ms

    if 'Preparing for pushing' in line:
        tokens=line.split(' ')
        num=tokens[-2] #sec
        print(tokens)
        prep.append(float(num)*1000) # make it ms

    if 'HandleSendFinished' in line:
        tokens=line.split(' ')
        num=tokens[-2] #sec
        print(tokens)
        send.append(float(num)*1000) # make it m

get_med = np.median(np.asarray(get))
#get_med = 0
put_med = np.median(np.asarray(put))
pull_med = np.median(np.asarray(pull))
push_med = np.median(np.asarray(push))
rest_med = np.median(np.asarray(rest))
prep_med = np.median(np.asarray(prep))
send_med = np.median(np.asarray(send))

transfer = sz_gb/2 *1000

#print("From Plasma (ms): ", plasma_med, " Deser (ms): ", deser_med, " Raylet (ms): ", ray_med)

print("Push median (ms):", push_med, "Push min (ms):", min(push), "Push max (ms):", max(push))
print("Pull (ms):", pull_med, "Pull min (ms):", min(pull), "Pull max (ms):", max(pull))

print("Put (ms): ", put_med, " Get (ms): ", get_med, " Transfer (ms): ", transfer, "Rest(ms): ", rest_med, "Prep (ms): ", prep_med)
'''

print("From Plasma (ms): ", plasma_med, " Deser (ms): ", deser_med, " Raylet (ms): ", ray_med)

meta = []
read_file = []
seal = []
create_buf = []

total_put = []

for name in ['restore_worker1.out', 'restore_worker2.out']:
    rs = open(name, 'r')
    for line in rs.readlines():
        if 'Reading metadata' in line:
            tokens=line.split(' ')
            num=tokens[-3] #sec
            print(tokens)
            meta.append(float(num)) # make it ms
        if 'read object from file' in line:
            tokens=line.split(' ')
            num=tokens[-3] #sec
            print(tokens)
            read_file.append(float(num)) # make it ms
        if 'Sealing' in line:
            tokens=line.split(' ')
            num=tokens[-3] #sec
            print(tokens)
            seal.append(float(num)) # make it ms
        if '_create_put_buffer' in line:
            tokens=line.split(' ')
            num=tokens[-3] #sec
            print(tokens)
            create_buf.append(float(num)) # make it ms
        if 'put_file_like_object' in line:
            tokens=line.split(' ')
            num=tokens[-3] #sec
            print(tokens)
            total_put.append(float(num)) # make it ms

meta_med = np.median(np.asarray(meta))
read_file_med = np.median(np.asarray(read_file))
seal_med = np.median(np.asarray(seal))
create_buf_med = np.median(np.asarray(create_buf))
total_put_med = np.median(np.asarray(total_put))

print("From Plasma (ms): ", plasma_med, " Deser (ms): ", deser_med, " Raylet (ms): ", ray_med)
print("Total put(ms): ", total_put_med, " meta: ", meta_med, " seal_med: ", seal_med, " create_buf_med: ", create_buf_med, " read_file_med: ", read_file_med)

'''