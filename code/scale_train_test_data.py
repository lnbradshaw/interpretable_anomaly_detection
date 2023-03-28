import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler

DATADIR = '../data/processed/train_test_data/'

print('Loading in train/test sets') # unfortunately going to have to do this manually, can't think of a better way to do so at the moment
with h5py.File(DATADIR+'qcd_dijet_train_test_data.h5', 'r') as hf:
	bkg = hf["train_test_data"][:]
with h5py.File(DATADIR+'h3_m174_h20_100k_01_train_test_data.h5', 'r') as hf:
	h20 = hf["train_test_data"][:]
with h5py.File(DATADIR+'h3_m174_h80_100k_01_train_test_data.h5', 'r') as hf:
	h80 = hf["train_test_data"][:]
with h5py.File(DATADIR+'top_100k_01_train_test_data.h5', 'r') as hf:
	top = hf["train_test_data"][:]
with h5py.File(DATADIR+'top_m80_100k_01_new_train_test_data.h5', 'r') as hf:
	t80 = hf["train_test_data"][:]
with h5py.File(DATADIR+'W_100k_01_train_test_data.h5', 'r') as hf:
	w = hf["train_test_data"][:]
with h5py.File(DATADIR+'W_m120_100k_01_train_test_data.h5', 'r') as hf:
	w120 = hf["train_test_data"][:]
with h5py.File(DATADIR+'W_m174_100k_01_train_test_data.h5', 'r') as hf:
	w174 = hf["train_test_data"][:]
with h5py.File(DATADIR+'W_m59_100k_01_train_test_data.h5', 'r') as hf:
	w59 = hf["train_test_data"][:]

bkg_train = bkg[:int(0.67*bkg.shape[0]),:]
bkg_test = bkg[int(0.67*bkg.shape[0]):,:]
#top_pt1200 = top_pt1200[:100000,:]
#t80_pt1200 = t80_pt1200[:100000,:]

SS = StandardScaler()
bkg_train = SS.fit_transform(bkg_train)
bkg_test = SS.transform(bkg_test)
h20 = SS.transform(h20)
h80 = SS.transform(h80)
#h20_pt1200 = SS.transform(h20_pt1200)
#h80_pt1200 = SS.transform(h80_pt1200)
#top_pt1200 = SS.transform(top_pt1200)
#t80_pt1200 = SS.transform(t80_pt1200)
#w_pt1200 = SS.transform(w_pt1200)
#w120_pt1200 = SS.transform(w120_pt1200)
#w174_pt1200 = SS.transform(w174_pt1200)
#w59_pt1200 = SS.transform(w59_pt1200)
top = SS.transform(top)
t80 = SS.transform(t80)
w = SS.transform(w)
w120 = SS.transform(w120)
w174 = SS.transform(w174)
w59 = SS.transform(w59)

if not os.path.exists(DATADIR+'preprocessed_and_scaled/'):
	os.makedirs(DATADIR+'preprocessed_and_scaled/')

with h5py.File(DATADIR+'preprocessed_and_scaled/bkg_train.h5', 'w') as hf:
	hf.create_dataset("data", data=bkg_train)
with h5py.File(DATADIR+'preprocessed_and_scaled/bkg_test.h5', 'w') as hf:
	hf.create_dataset("data", data=bkg_test)
with h5py.File(DATADIR+'preprocessed_and_scaled/h20_pt600.h5', 'w') as hf:
	hf.create_dataset("data", data=h20)
with h5py.File(DATADIR+'preprocessed_and_scaled/h80_pt600.h5', 'w') as hf:
	hf.create_dataset("data", data=h80)
with h5py.File(DATADIR+'preprocessed_and_scaled/top_pt600.h5', 'w') as hf:
	hf.create_dataset("data", data=top)
with h5py.File(DATADIR+'preprocessed_and_scaled/t80_pt600.h5', 'w') as hf:
	hf.create_dataset("data", data=t80)
with h5py.File(DATADIR+'preprocessed_and_scaled/w_pt600.h5', 'w') as hf:
	hf.create_dataset("data", data=w)
with h5py.File(DATADIR+'preprocessed_and_scaled/w120_pt600.h5', 'w') as hf:
	hf.create_dataset("data", data=w120)
with h5py.File(DATADIR+'preprocessed_and_scaled/w174_pt600.h5', 'w') as hf:
	hf.create_dataset("data", data=w174)
with h5py.File(DATADIR+'preprocessed_and_scaled/w59_pt600.h5', 'w') as hf:
	hf.create_dataset("data", data=w59)
