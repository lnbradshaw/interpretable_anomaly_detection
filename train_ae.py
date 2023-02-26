import os
import time
from sklearn.metrics import roc_curve, auc
import numpy as np
import tensorflow as tf
import energyflow as ef
import matplotlib.pyplot as plt
import h5py
from utils import make_roc, mse_img

imgdir = '../data/processed/images/'

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5, min_delta=1e-6) #, min_lr=1e-7)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=10, verbose=1, mode='auto', min_delta=1e-8)

with h5py.File(imgdir+'qcd_dijet_img.h5', 'r') as hf:
    bkg_img = hf["img"][:]
with h5py.File(imgdir+'W_100k_01_img.h5', 'r') as hf:
    w_img = hf["img"][:]
with h5py.File(imgdir+'top_100k_01_img.h5', 'r') as hf:
    t_img = hf["img"][:]
with h5py.File(imgdir+'h3_m174_h80_100k_01_img.h5', 'r') as hf:
    h_img = hf["img"][:]

bkg_train_img = bkg_img[:int(0.67*bkg_img.shape[0]), ...]
bkg_test_img = bkg_img[int(0.67*bkg_img.shape[0]):, ...]

print('Creating and training Convolutional Autoencoder')
latent_dim = 32
epochs = 50

generator = tf.keras.preprocessing.image.ImageDataGenerator()
train_ds = generator.flow(x=bkg_train_img, y=bkg_train_img, batch_size=256)
val_ds = generator.flow(x=bkg_test_img, y=bkg_test_img, batch_size=256)

input_img = tf.keras.Input(shape=(40,40,1))

layer = input_img
layer = tf.keras.layers.Conv2D(5, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Conv2D(5, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(layer)
layer = tf.keras.layers.Conv2D(5, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Conv2D(5, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Conv2D(5, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Conv2D(1, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Flatten()(layer)
layer = tf.keras.layers.Dense(100, activation='elu')(layer)
encoded = tf.keras.layers.Dense(latent_dim)(layer)

latent_inputs = tf.keras.Input(shape=(latent_dim,))
layer = tf.keras.layers.Dense(100, activation='elu')(latent_inputs)
layer = tf.keras.layers.Dense(400, activation='elu')(layer)
layer = tf.keras.layers.Reshape((20,20,1))(layer)
layer = tf.keras.layers.Conv2D(5, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Conv2D(5, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Conv2DTranspose(5, kernel_size=(2,2), strides=(2,2), padding='same')(layer)
layer = tf.keras.layers.Conv2D(5, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Conv2D(1, kernel_size=(3,3), activation='elu', padding='same')(layer)
layer = tf.keras.layers.Reshape((1,1600))(layer)
layer = tf.keras.layers.Activation('softmax')(layer)
decoded = tf.keras.layers.Reshape((40,40,1))(layer)

encoder = tf.keras.Model(input_img, encoded, name='encoder')
decoder = tf.keras.Model(latent_inputs, decoded, name='decoder')
autoencoder = tf.keras.Model(input_img, decoder(encoder(input_img)))

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
print('')
print(autoencoder.summary())
print('')
autoencoder.fit(train_ds, epochs=epochs, shuffle=True, validation_data=(val_ds), callbacks=[es, reduce_lr], verbose=1)
print('Autoencoder trained, getting predictions')

bkg_preds = decoder.predict(encoder.predict(bkg_test_img))
w_preds = decoder.predict(encoder.predict(w_img))
t_preds = decoder.predict(encoder.predict(t_img))
h_preds = decoder.predict(encoder.predict(h_img))

bkg_mse = mse_img(bkg_test_img, bkg_preds)
w_mse = mse_img(w_img, w_preds)
t_mse = mse_img(t_img, t_preds)
h_mse = mse_img(h_img, h_preds)

w_tpr, w_fpr, w_auc, w_es10, w_es100 = make_roc(w_mse, bkg_mse)
t_tpr, t_fpr, t_auc, t_es10, t_es100 = make_roc(t_mse, bkg_mse)
h_tpr, h_fpr, h_auc, h_es10, h_es100 = make_roc(h_mse, bkg_mse)

if not os.path.exists('../models/'):
    os.makedirs('../models/')
tf.keras.models.save_model(encoder,'../models/encoder')
tf.keras.models.save_model(decoder,'../models/decoder')

#np.save('w_tpr_ae.npy',w_tpr)
#np.save('w_fpr_ae.npy',w_fpr)
#np.save('t_tpr_ae.npy',t_tpr)
#np.save('t_fpr_ae.npy',t_fpr)
#np.save('h_tpr_ae.npy',h_tpr)
#np.save('h_fpr_ae.npy',h_fpr)

print(f"W AUC: {np.round(w_auc, 3)}") #", {np.round(w_es10, 4)}, {np.round(w_es100,4)}")
print(f"Top AUC: {np.round(t_auc, 3)}") #", {np.round(t_es10, 4)}, {np.round(t_es100,4)}")
print(f"Higgs AUC: {np.round(h_auc, 3)}") #", {np.round(h_es10, 4)}, {np.round(h_es100,4)}")

fig, ax = plt.subplots(1,1,figsize=(4,4),constrained_layout=True)
ax.plot(w_tpr, 1/w_fpr, color='tab:orange')
ax.plot(t_tpr, 1/t_fpr, color='tab:green')
ax.plot(h_tpr, 1/h_fpr, color='tab:red')
ax.scatter([],[],color='tab:orange',label='W AUC: {:.3f}'.format(w_auc))
ax.scatter([],[],color='tab:green',label='Top AUC: {:.3f}'.format(t_auc))
ax.scatter([],[],color='tab:red',label='Higgs AUC: {:.3f}'.format(h_auc))
ax.legend(loc='best')
ax.set_yscale('log')
ax.grid()
ax.set_xticks([0.00,0.25,0.50,0.75,1.00])
ax.set_ylim(9e-1,1e3)

plt.show()
