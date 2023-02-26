bkg_url = 'https://zenodo.org/record/4641460/files/qcd_dijet.h5'

print(bkg_url.rsplit('/')[-1])
print(bkg_url.rsplit('/')[:-1])


import tqdm as tqdm

import numpy as np
import random
import itertools
from time import perf_counter
from utils import *

def get_pairs(arr,num_events):
	pairs = list(itertools.combinations(list(arr[:num_events,...]),2))
	p1,p2 = zip(*pairs)
	p1 = np.array(p1)
	p2 = np.array(p2)
	del pairs

	return p1,p2

def get_random_indices(arr,num):
	random_indices = random.sample(range(0, arr.shape[0], 1), num)

	return random_indices

#test = np.random.random((int(5e7),1))
#print(test.shape)
#test.itemsize*test.shape[0]
#print(test.dtype)

# need to make a generator to save memory (and run things locally)
def deco(arr1, arr2):
    arr1 = arr1.reshape(-1,1)
    arr2 = arr2.reshape(-1,1)
    val = np.heaviside(arr1*arr2,1)
    #val = np.asarray(val)
    #val = val.reshape(-1,)
    return val

random_indices = random.sample(range(0,int(1e5),1),int(1e4))
#print(random_indices)
batch_size = int(1e3)
array1 = np.asarray(random.sample(range(0,int(1e6),1),int(1e5)))
array1 = array1.reshape(-1,1)
array2 = np.asarray(random.sample(range(0,int(1e6),1),int(1e5)))
array2 = array2.reshape(-1,1)
#print(array1.shape)

meh = make_roc(array1,array2)
print(meh[0].shape)
print(meh[3])

#print(len(list(itertools.combinations(random_indices,2))))
#l = len(random_indices)
#bs = 10
#for i in range(bs):
#    print(len(random_indices[int(i*l/bs):int((i+1)*l/bs)]))

#p0,p1 = zip(*list(itertools.combinations(random_indices,2)))
#print(len(p0))
#p0 = np.asarray(p0)
#p0  = p0.reshape(-1,1)
#print(p0.shape)
#print(array1[p0,:].shape)

def ado_batch_generator(arr1,arr2,random_indices,batch_size=int(1e3)):
	do_tmp = []
	p0,p1 = zip(*list(itertools.combinations(random_indices,2)))
	nbatch = int(len(p0)/batch_size)
	print(nbatch)
	for i in range(nbatch):
		p00 = p0[int(i*len(p0)/nbatch):int((i+1)*len(p0)/nbatch)]
		p11 = p1[int(i*len(p1)/nbatch):int((i+1)*len(p1)/nbatch)]
		if i % 500 == 0:
			print(i)
			print(len(p00))
        #print(f"Iteration {i}")
		d1 = arr1[p00,:]-arr1[p11,:]
		d2 = arr2[p00,:]-arr2[p11,:]
		do_tmp.append(deco(d1,d2))
        #val = deco(d1,d2)
	yield do_tmp
start = perf_counter()
#do0 = ado_batch_generator(array1,array2,random_indices,int(5e3))
deco0 = next(ado_batch_generator(array1,array2,random_indices,int(4.5e4)))
e1 = perf_counter()
print(f"Getting the DO took {np.round(e1-start,4)} s")
#deco1 = next(do0)
#print(deco0==deco1)
print(len(deco0))
print(len(deco0[0]))
decoo = [item for sublist in deco0 for item in sublist]
#decoo = list(itertools.chain.from_iterable(deco0))
#print(len(decoo))
#decoo = []
#for sublist in deco0:
#    for item in sublist:
#        decoo.append(item)
print(len(decoo))
e2 = perf_counter()
print(f"Unpacking the list took {np.round(e2-e1,4)} s")
dv = np.asarray(decoo)
print(dv.shape)
dv = dv.reshape(-1,)
print(np.mean(dv))

meh = dv[dv==0]
print(meh.shape)

p0,p1 = zip(*list(itertools.combinations(random_indices,2)))
p00 = np.array(p0)[dv==0]
print(p00.shape)
p11 = np.array(p1)[dv==0]
#print(dv[:15])
#print(np.mean(np.reshape(np.asarray(decoo),(-1,1))))
end = perf_counter()
print(f"This took {np.round(end-start)} seconds to run.")
