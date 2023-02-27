# start by loading in necessary packages

import os
import time

start=time.time()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import pandas as pd
import itertools
import tensorflow as tf
import h5py
import random
import itertools
#from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
#from scipy import interpolate
#from matplotlib import ticker, cm
#from sklearn.preprocessing import LabelEncoder
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import energyflow as ef
from utils import *

plt.rcParams.update({'font.family': 'cmr10',
                     'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

print('Packages imported')

DATADIR = '../data/'
MODELDIR = '../models/'
latent_dim = 32

def main():
    with h5py.File(DATADIR+'processed/images/qcd_dijet_img.h5', 'r') as hf:
        bkg_img = hf["img"][:]
    with h5py.File(DATADIR+'processed/images/W_100k_01_img.h5', 'r') as hf:
        w_img = hf["img"][:]
    with h5py.File(DATADIR+'processed/images/top_100k_01_img.h5', 'r') as hf:
        t_img = hf["img"][:]
    with h5py.File(DATADIR+'processed/images/h3_m174_h80_100k_01_img.h5', 'r') as hf:
        h_img = hf["img"][:]
    bkg_train_img = bkg_img[:int(0.67*bkg_img.shape[0]),...]
    bkg_test_img = bkg_img[int(0.67*bkg_img.shape[0]):,...]

    with h5py.File(DATADIR+'processed/train_test_data/preprocessed_and_scaled/bkg_test.h5', 'r') as hf:
        bkg_test = hf["data"][:]
    with h5py.File(DATADIR+'processed/train_test_data/preprocessed_and_scaled/bkg_train.h5', 'r') as hf:
        bkg_train = hf["data"][:]
    with h5py.File(DATADIR+'processed/train_test_data/preprocessed_and_scaled/w_pt600.h5', 'r') as hf:
        w_test = hf["data"][:]
    with h5py.File(DATADIR+'processed/train_test_data/preprocessed_and_scaled/top_pt600.h5', 'r') as hf:
        t_test = hf["data"][:]
    with h5py.File(DATADIR+'processed/train_test_data/preprocessed_and_scaled/h80_pt600.h5', 'r') as hf:
        h_test = hf["data"][:]
    print('Data loaded')

    print('Getting CNN predictions')
    encoder = tf.keras.models.load_model(MODELDIR+'encoder')
#encoder.compile()
    decoder = tf.keras.models.load_model(MODELDIR+'decoder')
#decoder.compile()

# get target output for pnns
    encoded_rep_bkg_train = encoder.predict(bkg_train_img)
    encoded_rep_bkg_test = encoder.predict(bkg_test_img)
    encoded_rep_w = encoder.predict(w_img)
    encoded_rep_t = encoder.predict(t_img)
    encoded_rep_h = encoder.predict(h_img)

    tr_bkg_mse = mse_img(bkg_train_img,decoder.predict(encoded_rep_bkg_train))
    tr_bkg_mse = tr_bkg_mse.reshape(-1,1)

    target_output = tr_bkg_mse[int(np.floor(tr_bkg_mse.shape[0]/2)):,:]-tr_bkg_mse[:int(np.ceil(tr_bkg_mse.shape[0]/2)),:]

    target_output[target_output > 0] = 1
    target_output[target_output <= 0] = 0
#print(target_output.shape)

    bkg_preds = decoder.predict(encoded_rep_bkg_test)
    w_preds = decoder.predict(encoded_rep_w)
    t_preds = decoder.predict(encoded_rep_t)
    h_preds = decoder.predict(encoded_rep_h)
#print(bkg_preds.shape)

    bkg_mse_img = mse_img(bkg_test_img, bkg_preds)
    w_mse_img = mse_img(w_img, w_preds)
    t_mse_img = mse_img(t_img, t_preds)
    h_mse_img = mse_img(h_img, h_preds)

    del bkg_train_img, bkg_test_img, w_img, t_img, h_img
    del bkg_preds, w_preds, t_preds, h_preds

    w_tpr, w_fpr, w_auc = make_roc(w_mse_img, bkg_mse_img)
    t_tpr, t_fpr, t_auc = make_roc(t_mse_img, bkg_mse_img)
    h_tpr, h_fpr, h_auc = make_roc(h_mse_img, bkg_mse_img)
    print('')
    print('W AUC: {:.4f}'.format(w_auc))
    print('Top AUC: {:.4f}'.format(t_auc))
    print('Higgs AUC: {:.4f}'.format(h_auc))
    print('')
    print('Have CNN predictions')

    num_events = 10000

    if os.path.exists(DATADIR+'processed/random_indices.py'):
        random_indices = np.load(DATADIR+'processed/random_indices.npy')
    if not os.path.exists(DATADIR+'processed/random_indices.py'):
        random_indices = get_random_indices(tr_bkg_mse, num_events)
        np.save(DATADIR+'processed/random_indices.npy',random_indices)

    index_skip_list = []
    obs_ado_list = []
    model_ado_list = []
    w_auc_list = []
    t_auc_list = []
    h_auc_list = []
    bkg_mse_list = []

    pnn0_tr_bkg = bkg_train[:,:2]
    pnn0_tr1 = bkg_train[:int(np.ceil(bkg_train.shape[0]/2)),:2]
    pnn0_tr2 = bkg_train[int(np.floor(bkg_train.shape[0]/2)):,:2]
    pnn0_te_bkg = bkg_test[:,:2]
    pnn0_te_w = w_test[:,:2]
    pnn0_te_t = t_test[:,:2]
    pnn0_te_h = h_test[:,:2]
    bkg_train_efps = bkg_train[:,2:]
    bkg_test_efps = bkg_test[:,2:]
    w_test_efps = w_test[:,2:]
    t_test_efps = t_test[:,2:]
    h_test_efps = h_test[:,2:]
#print(pnn0_tr_bkg.shape)

    print('Training PNN on mass and pt')
    pnn0 = make_paired_nn(pnn0_tr1,pnn0_tr2,target_output=target_output,epochs=200,initializer='glorot_normal')
    print('Model trained, computing ADO')
#pnn0_int = tf.keras.models.load_model(MODELDIR+'pnn0_int_model')

#pair0,pair1 = zip(*list(itertools.combinations(random_indices,2)))
#p0_pred = pnn0_int.predict(pnn0_tr_bkg[pair0,:])
#p1_pred = pnn0_int.predict(pnn0_tr_bkg[pair1,:])
#cnn_mse_dif = tr_bkg_mse[pair0,:] - tr_bkg_mse[pair1,:]
#cnn_mse_dif = cnn_mse_dif.reshape(-1,1)
#pnn_dif = p0_pred - p1_pred
#pnn_dif = pnn_dif.reshape(-1,1)
#pnn0_do = np.heaviside(cnn_mse_dif*pnn_dif,1)
#pnn0_do = np.array(pnn0_do)
#pnn0_do = pnn0_do.reshape(-1,)
    pnn0_do = get_do(pnn0[1].predict(pnn0_tr_bkg),tr_bkg_mse,random_indices,prev_do=None)
    pnn0_ado = np.mean(pnn0_do)
#del p0_pred, p1_pred, pnn_dif

    w_tpr, w_fpr, w_auc_tmp = make_roc(pnn0[1].predict(pnn0_te_w), pnn0[1].predict(pnn0_te_bkg))
    t_tpr, t_fpr, t_auc_tmp = make_roc(pnn0[1].predict(pnn0_te_t), pnn0[1].predict(pnn0_te_bkg))
    h_tpr, h_fpr, h_auc_tmp = make_roc(pnn0[1].predict(pnn0_te_h), pnn0[1].predict(pnn0_te_bkg))
    w_auc_list.append(w_auc_tmp)
    t_auc_list.append(t_auc_tmp)
    h_auc_list.append(h_auc_tmp)
    model_ado_list.append(pnn0_ado)
    if not os.path.exists(MODELDIR+'pnn_int_models/'):
        os.makedirs(MODELDIR+'pnn_int_models/')
    pnn0[1].save(MODELDIR+'pnn_int_models/pnn0_int_model')

    print('PNN0 ADO: {:.4f}, W AUC: {:.4f}, Top AUC: {:.4f}, Higgs AUC {:.4f}'.format(pnn0_ado, w_auc_tmp, t_auc_tmp, h_auc_tmp))

    for i in range(15):
        print('Iteration: '+str(i))
        if len(index_skip_list)==0:
        # ADO loop to get best initial observable
            tmp_ado_list = []
            for j in range(bkg_train_efps.shape[1]):
            #efp_dif = bkg_train_efps[pair0,j] - bkg_train_efps[pair1,j]
            #efp_dif = efp_dif.reshape(-1,1)
            #do_tmp = np.heaviside(cnn_mse_dif*efp_dif,1)
            #do_tmp = do_tmp[pnn0_do==0]
                do_tmp = get_do(bkg_train_efps[:,j],tr_bkg_mse,random_indices,prev_do=pnn0_do)
            #do_tmp = do_tmp[pnn0_do==0]
                ado_tmp = np.mean(do_tmp)
                if ado_tmp < 0.5:
                    ado_tmp = 1-ado_tmp
                print('Obs. {} correctly orders {}/{} pairs misordered by PNN {}'.format(j,do_tmp[do_tmp==1].shape[0],do_tmp.shape[0],i))
                del do_tmp
                tmp_ado_list.append(ado_tmp)

            for k in range(len(tmp_ado_list)):
                if list_contains(index_skip_list,[k]):
                    tmp_ado_list[k]=0

            new_index = tmp_ado_list.index(np.max(tmp_ado_list))
            index_skip_list.append(new_index)
            print('     Best Initial Observable {} ADO {:.4f}'.format(new_index,np.max(tmp_ado_list)))
            obs_ado_list.append(np.max(tmp_ado_list))
		#model_ado_list.append(np.max(tmp_ado_list))
		#bkg_mse_list.append(bkg_test_efp[:,new_index])
		#tpr,fpr,auc_tmp = make_roc(sig_test_efp[:,new_index],bkg_test_efp[:,new_index])
		#auc_list.append(auc_tmp)
            del pnn0_do

            print('Indices to skip: '+str(index_skip_list))
            print('')


        else:
        # loop to construct PNNs and find subsequent best observables
            print('     Creating observable array')
            pnn_tr = np.zeros((bkg_train_efps.shape[0], len(index_skip_list)))
            pnn_te = np.zeros((bkg_test_efps.shape[0], len(index_skip_list)))
            pnn_w_te = np.zeros((w_test_efps.shape[0], len(index_skip_list)))
            pnn_t_te = np.zeros((t_test_efps.shape[0], len(index_skip_list)))
            pnn_h_te = np.zeros((h_test_efps.shape[0], len(index_skip_list)))
            for p,obs in enumerate(index_skip_list):
                pnn_tr[:,p] = bkg_train_efps[:,obs]
                pnn_te[:,p] = bkg_test_efps[:,obs]
                pnn_w_te[:,p] = w_test_efps[:,obs]
                pnn_t_te[:,p] = t_test_efps[:,obs]
                pnn_h_te[:,p] = h_test_efps[:,obs]
            pnn_tr = np.hstack((pnn0_tr_bkg, pnn_tr))
            pnn_te = np.hstack((pnn0_te_bkg, pnn_te))
            pnn_w_te = np.hstack((pnn0_te_w, pnn_w_te))
            pnn_t_te = np.hstack((pnn0_te_t, pnn_t_te))
            pnn_h_te = np.hstack((pnn0_te_h, pnn_h_te))

            print('     Training PNN {}'.format(i))
            pnn = make_paired_nn(pnn_tr[:int(np.ceil(pnn_tr.shape[0]/2)),:],pnn_tr[int(np.floor(pnn_tr.shape[0]/2)):,:],target_output=target_output,epochs=200,initializer='glorot_normal')
            pnn[1].save(MODELDIR+'pnn_int_models/pnn{}_int'.format(i))
            print('     Computing PNN {} ADO'.format(i))
		#p0_preds = pnn[1].predict(pnn_tr[pair0,:])
		#p1_preds = pnn[1].predict(pnn_tr[pair1,:])
		#pnn_dif = p0_preds - p1_preds
		#pnn_dif = pnn_dif.reshape(-1,1)
		#pnn_do = np.heaviside(cnn_mse_dif*pnn_dif,1)
		#pnn_do = np.array(pnn_do).reshape(-1,)
            pnn_do = get_do(pnn[1].predict(pnn_tr),tr_bkg_mse,random_indices,prev_do=None)
            pnn_ado = np.mean(pnn_do)
            print('     PNN {} ADO: {:.4f}'.format(i,pnn_ado))
            model_ado_list.append(pnn_ado)
            w_preds = pnn[1].predict(pnn_w_te)
            t_preds = pnn[1].predict(pnn_t_te)
            h_preds = pnn[1].predict(pnn_h_te)
            bkg_preds = pnn[1].predict(pnn_te)
            w_tpr, w_fpr, w_auc_tmp = make_roc(w_preds, bkg_preds)
            t_tpr, t_fpr, t_auc_tmp = make_roc(t_preds, bkg_preds)
            h_tpr, h_fpr, h_auc_tmp = make_roc(h_preds, bkg_preds)
            w_auc_list.append(w_auc_tmp)
            t_auc_list.append(t_auc_tmp)
            h_auc_list.append(h_auc_tmp)
            print('     PNN {} W AUC: {:.4f}, Top AUC: {:.4f}, Higgs AUC {:.4f}'.format(i, w_auc_tmp, t_auc_tmp, h_auc_tmp))
            print('     Finding next best observable')
            tmp_ado_list = []
            for j in range(bkg_train_efps.shape[1]):
			#efp_dif  = bkg_train_efps[pair0,j] - bkg_train_efps[pair1,j]
			#efp_dif = efp_dif.reshape(-1,1)
			#do_tmp = np.heaviside(cnn_mse_dif*efp_dif,1)
                do_tmp = get_do(bkg_train_efps[:,j],tr_bkg_mse,random_indices,prev_do=pnn_do)
            #do_tmp  = do_tmp[pnn_do==0]
                ado_tmp = np.mean(do_tmp)
                if ado_tmp < 0.5:
                    ado_tmp = 1-ado_tmp
                print('Obs. {} correctly orders {}/{} pairs misordered by PNN {}.'.format(j, do_tmp[do_tmp==1].shape[0], do_tmp.shape[0], i))
                #print('Obs. {} has an ADO of {}.'.format(j, ado_tmp))
                del do_tmp
                tmp_ado_list.append(ado_tmp)

            for k in range(len(tmp_ado_list)):
			#if np.isnan(tmp_ado_list[k]):
			#	tmp_ado_list[k] == 0
                if list_contains(index_skip_list, [k]):
                    tmp_ado_list[k] = 0

		#print(tmp_ado_list)

            new_index = tmp_ado_list.index(np.max(tmp_ado_list))
            index_skip_list.append(new_index)
            obs_ado_list.append(np.max(tmp_ado_list))
            del pnn_do

            print('Indices to skip: '+str(index_skip_list))
            print('')


    print('Model ADOs: ' + str(model_ado_list))
    print('W AUCs: ' + str(w_auc_list))
    print('Top AUCs: ' + str(t_auc_list))
    print('Higgs AUCs: ' + str(h_auc_list))
    end=time.time()
    print('This script took {:.3f} seconds to run'.format(end-start))

    np.save(DATADIR+'processed/pnn_indices_15.npy',index_skip_list)
#np.save(DATADIR+'processed/pnn_auc_list_15.npy',auc_list)
    np.save(DATADIR+'processed/pnn_ado_list_15.npy',model_ado_list)

if __name__ == "__main__":
    main()
