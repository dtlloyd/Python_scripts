# Use Bubble sort algorithm to sort a numerical list
# test the algorithm execution time vs array length
import time
import random
import matplotlib.pyplot as plt
import numpy as np

ar_length = np.array([100,250,500,1000,2500,5000,10000,25000])
n_lengths = np.shape(ar_length)

def bub_sort(random_list):
    test=1
    while test==1:
        test=0
        for i in range(0,len(random_list)-1):
            if random_list[i+1]<random_list[i]:
                random_list[i+1],random_list[i]=random_list[i],random_list[i+1]
                test=1
    return random_list

t=np.array([0,0,0,0,0,0,0,0])
ts=np.array([0,0,0,0,0,0,0,0])

for j in range(0,n_lengths[0]):
    top_num=100*ar_length[n_lengths[0]-1] #largest possible number in random list
    list_num=ar_length[j] #number of entries in random list
    r_l = random.sample(range(top_num), list_num) #create a list of random integers
    start_time = time.time()   
    bub_sort(r_l)
    t[j]=time.time() - start_time
    ts[j]=np.sqrt(t[j])
plt.plot(ar_length,ts)
plt.xlabel('Array length')
plt.ylabel('Time taken to sort (s)')