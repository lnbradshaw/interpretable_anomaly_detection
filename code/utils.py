import numpy as np
import tensorflow as tf
import itertools
from sklearn.metrics import roc_curve, auc
import random

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1,
                                  patience=5, min_lr=1.0e-6)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto',min_delta=1e-4)


def make_roc(sig_score, bkg_score):
    sig_score = sig_score.reshape(-1,1)
    bkg_score = bkg_score.reshape(-1,1)
    bkg_true = np.zeros_like((bkg_score))
    sig_true = np.ones_like((sig_score))
    fpr, tpr, thresh = roc_curve(np.vstack((bkg_true, sig_true)), np.vstack((bkg_score, sig_score)))
    aucs = auc(fpr, tpr)
    #eb50 = fpr[np.argmin(np.abs(tpr-0.5))]
    #eb10 = fpr[np.argmin(np.abs(tpr-0.1))]
    #es10 = tpr[np.argmin(np.abs((1/fpr)-10))]
    #es100 = tpr[np.argmin(np.abs((1/fpr)-100))]

    return tpr, fpr, aucs #, 1/eb50, 1/eb10

def mse_img (img1, img2):
    assert img1.shape == img2.shape
    squared_dif = np.square(img1 - img2)
    summed1 = np.sum(squared_dif, axis=2)
    summed2 = np.sum(summed1, axis=1)
    num_pix = img1.shape[1]*img1.shape[2]
    err = summed2/num_pix

    return err

def make_paired_nn(input_data1,input_data2,target_output,epochs,initializer='glorot_normal'):

	assert input_data1.shape[1]==input_data2.shape[1]

	gen_in = tf.keras.Input(shape=(input_data1.shape[1]))
	l1 = tf.keras.layers.Dense(50,activation='elu',kernel_initializer=initializer)(gen_in)
	l2 = tf.keras.layers.Dense(50,activation='elu',kernel_initializer=initializer)(l1)
	l3 = tf.keras.layers.Dense(50,activation='elu',kernel_initializer=initializer)(l2)
	l4 = tf.keras.layers.Dense(50,activation='elu',kernel_initializer=initializer)(l3)
	gen_out = tf.keras.layers.Dense(1,activation='relu',kernel_initializer=initializer)(l4)

	int_model = tf.keras.Model(inputs=gen_in,outputs=gen_out)

	i1 = tf.keras.Input(shape=(input_data1.shape[1],))
	i2 = tf.keras.Input(shape=(input_data2.shape[1],))
	subtracted = tf.keras.layers.Subtract()([int_model(i2),int_model(i1)])

	output = tf.keras.activations.sigmoid(subtracted)

	#instantiate model
	model = tf.keras.Model(inputs=[i1,i2],outputs=output)

	#compile and train model
	model.compile(optimizer='adam',loss='binary_crossentropy')
	model.fit([input_data1,input_data2],target_output,epochs=epochs,verbose=2,batch_size=256,callbacks=[reduce_lr,es],validation_split=0.15)

	return model,int_model

def make_high_level_nn(input_data,target_output,epochs,initializer='glorot_normal'):

	inputs = tf.keras.Input(shape=(input_data.shape[1]))
	l1 = tf.keras.layers.Dense(50,activation='elu',kernel_initializer=initializer)(inputs)
	l2 = tf.keras.layers.Dense(50,activation='elu',kernel_initializer=initializer)(l1)
	l3 = tf.keras.layers.Dense(50,activation='elu',kernel_initializer=initializer)(l2)
	l4 = tf.keras.layers.Dense(50,activation='elu',kernel_initializer=initializer)(l3)
	outputs = tf.keras.layers.Dense(1,activation='linear',kernel_initializer=initializer)(l4)

	#instantiate model
	model = tf.keras.Model(inputs=inputs, outputs=outputs)

	#compile and train model
	model.compile(optimizer='adam',loss='mse')
	model.fit(input_data,target_output,epochs=epochs,verbose=2,batch_size=256,callbacks=[reduce_lr,es],validation_split=0.15)

	return model

def list_contains(List1, List2):
	check = False
	# Iterate in the 1st list
	for m in List1:
        # Iterate in the 2nd list
		for n in List2:
		# if there is a match
			if m == n:
				check = True

	return check

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

def deco(arr1, arr2):
    val = np.heaviside(arr1*arr2,1)
    return val

def do_batch_generator(arr1,arr2,random_indices,prev_do,batch_size=int(1e6)):
    arr1 = arr1.reshape(-1,1)
    arr2 = arr2.reshape(-1,1)
    do_tmp = []
    p0,p1 = zip(*list(itertools.combinations(random_indices,2)))
    if prev_do is not None:
        p0 = list(np.array(p0)[prev_do==0])
        p1 = list(np.array(p1)[prev_do==0])
    nbatch = int(len(p0)/batch_size)
    for i in range(nbatch):
        p00 = p0[int(i*len(p0)/nbatch):int((i+1)*len(p0)/nbatch)]
        p11 = p1[int(i*len(p1)/nbatch):int((i+1)*len(p1)/nbatch)]
        #print(f"Iteration {i}")
        d1 = arr1[p00,:]-arr1[p11,:]
        d2 = arr2[p00,:]-arr2[p11,:]
        do_tmp.append(deco(d1,d2))
        #val = deco(d1,d2)
    yield do_tmp

def get_do(arr1,arr2,random_indices,prev_do):
    dol = next(do_batch_generator(arr1,arr2,random_indices,prev_do))
    dol = [item for sublist in dol for item in sublist]
    dol = np.asarray(dol)
    dol = dol.reshape(-1,)
    return dol
