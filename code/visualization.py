import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

plt.rcParams.update({'font.family':'cmr10', 'font.size': 12})
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 80
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.formatter.use_mathtext'] = True

# start with visualizing the jet images
imgdir = '../data/processed/images/'

with h5py.File(imgdir+'qcd_dijet_img.h5', 'r') as hf:
    bkg_img = hf["img"][:]
with h5py.File(imgdir+'W_100k_01_img.h5', 'r') as hf:
    w_img = hf["img"][:]
with h5py.File(imgdir+'top_100k_01_img.h5', 'r') as hf:
    top_img = hf["img"][:]
with h5py.File(imgdir+'h3_m174_h80_100k_01_img.h5', 'r') as hf:
    higgs_img = hf["img"][:]
num_jet = min(bkg_img.shape[0], w_img.shape[0], top_img.shape[0], higgs_img.shape[0])

print(min(np.mean(bkg_img[:num_jet,...],axis=0)[:,:,0].min(), np.mean(w_img[:num_jet,...],axis=0)[:,:,0].min(),
np.mean(top_img[:num_jet,...],axis=0)[:,:,0].min(), np.mean(higgs_img[:num_jet,...],axis=0)[:,:,0].min()),
max(np.mean(bkg_img[:num_jet,...],axis=0)[:,:,0].max(), np.mean(w_img[:num_jet,...],axis=0)[:,:,0].max(),
np.mean(top_img[:num_jet,...],axis=0)[:,:,0].max(), np.mean(higgs_img[:num_jet,...],axis=0)[:,:,0].max()))

vmin = 1e-3
vmax = 1e-1

xticks = [-1.00,-0.50,0.00,-0.50,1.00]
yticks = [-1.00,-0.50,0.00,0.50,1.00]

fig, ax = plt.subplots(2, 2, figsize=(7,6), constrained_layout=True)
#norm = LogNorm(min(np.mean(bkg_img,axis=0)[:,:,0].min(), np.mean(w_img,axis=0)[:,:,0].min(),
#np.mean(top_img,axis=0)[:,:,0].min(), np.mean(higgs_img,axis=0)[:,:,0].min()),
#max(np.mean(bkg_img,axis=0)[:,:,0].max(), np.mean(w_img,axis=0)[:,:,0].max(),
#np.mean(top_img,axis=0)[:,:,0].max(), np.mean(higgs_img,axis=0)[:,:,0].max()))
print(bkg_img.shape, w_img.shape, top_img.shape, higgs_img.shape)

im0 = ax[0,0].pcolormesh(np.mean(bkg_img[:num_jet,...],axis=0)[:,:,0],cmap='jet',norm=LogNorm(vmin=vmin,vmax=vmax))
ax[0,0].set_xticklabels(xticks)
ax[0,0].set_yticklabels(yticks)
ax[0,0].set_ylabel(r'$\eta$', fontsize='medium')
ax[0,0].set_xlabel(r'$\phi$', fontsize='medium')
ax[0,0].set_title('Background',fontsize='x-large')
#ax[0,0].set_aspect('equal')

im1 = ax[0,1].pcolormesh(np.mean(w_img[:num_jet,...],axis=0)[:,:,0],cmap='jet',norm=LogNorm(vmin=vmin,vmax=vmax))
ax[0,1].set_xticklabels(xticks)
ax[0,1].set_yticklabels(yticks)
ax[0,1].set_ylabel(r'$\eta$', fontsize='medium')
ax[0,1].set_xlabel(r'$\phi$', fontsize='medium')
ax[0,1].set_title(r'80 GeV $W$', fontsize='x-large')
#ax[0,1].set_aspect('equal')

im2 = ax[1,0].pcolormesh(np.mean(top_img[:num_jet,...],axis=0)[:,:,0],cmap='jet',norm=LogNorm(vmin=vmin,vmax=vmax))
ax[1,0].set_xticklabels(xticks)
ax[1,0].set_yticklabels(yticks)
ax[1,0].set_ylabel(r'$\eta$', fontsize='medium')
ax[1,0].set_xlabel(r'$\phi$', fontsize='medium')
ax[1,0].set_title('174 GeV Top', fontsize='x-large')
#ax[1,0].set_aspect('equal')

im3 = ax[1,1].pcolormesh(np.mean(higgs_img[:num_jet,...],axis=0)[:,:,0],cmap='jet',norm=LogNorm(vmin=vmin,vmax=vmax))
ax[1,1].set_xticklabels(xticks)
ax[1,1].set_yticklabels(yticks)
ax[1,1].set_ylabel(r'$\eta$', fontsize='medium')
ax[1,1].set_xlabel(r'$\phi$', fontsize='medium')
ax[1,1].set_title('80 GeV Higgs', fontsize='x-large')
#ax[1,1].set_aspect('equal')

#fig.subplots_adjust(right=0.85)
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
#fig.colorbar(im0, cax=cbar_ax)

cbar = fig.colorbar(im0, ax=ax.ravel().tolist(), extend='both')
cbar.set_label(r'Normalized $p_{T}$',fontsize='x-large')

plt.show()

#plt.savefig('../plots/avg_jet_img_v2.pdf', bbox_inches='tight')

# print out pt range for bkg, each signal
with h5py.File('../data/processed/train_test_data/qcd_dijet_train_test_data.h5', 'r') as hf:
    bkg_data = hf["train_test_data"][:]
with h5py.File('../data/processed/train_test_data/W_100k_01_train_test_data.h5', 'r') as hf:
    w_data = hf["train_test_data"][:]
with h5py.File('../data/processed/train_test_data/top_100k_01_train_test_data.h5', 'r') as hf:
    top_data = hf["train_test_data"][:]
with h5py.File('../data/processed/train_test_data/h3_m174_h80_100k_01_train_test_data.h5', 'r') as hf:
    h80_data = hf["train_test_data"][:]

print(f"Min Bkg. pt: {min(bkg_data[:,1])} / Max Bkg. pt: {max(bkg_data[:,1])}")
print(f"Min W pt: {min(w_data[:,1])} / Max W pt: {max(w_data[:,1])}")
print(f"Min Top pt: {min(top_data[:,1])} / Max Top pt: {max(top_data[:,1])}")
print(f"Min Higgs pt: {min(h80_data[:,1])} / Max Higgs pt: {max(h80_data[:,1])}")

min_pt = 450
max_pt = 2500
print(bkg_data.shape)
print(w_data.shape)
print(top_data.shape)
print(h80_data.shape)
print('')
bkg_data = bkg_data[bkg_data[:,1]>=min_pt]
w_data = w_data[w_data[:,1]>=min_pt]
top_data = top_data[top_data[:,1]>=min_pt]
h80_data = h80_data[h80_data[:,1]>=min_pt]
print(bkg_data.shape)
print(w_data.shape)
print(top_data.shape)
print(h80_data.shape)
print('')
bkg_data = bkg_data[bkg_data[:,1]<=max_pt]
w_data = w_data[w_data[:,1]<=max_pt]
top_data = top_data[top_data[:,1]<=max_pt]
h80_data = h80_data[h80_data[:,1]<=max_pt]
print(bkg_data.shape)
print(w_data.shape)
print(top_data.shape)
print(h80_data.shape)
print('')

min_pt = min(min(bkg_data[:,1]), min(w_data[:,1]), min(top_data[:,1]), min(h80_data[:,1]))
print(min_pt)
max_pt = max(max(bkg_data[:,1]), max(w_data[:,1]), max(top_data[:,1]), max(h80_data[:,1]))
print(max_pt)
print(bkg_data[:,1].shape)
fig, ax = plt.subplots(1, 1, figsize=(4,4), constrained_layout=True)

pt_bins = np.linspace(min_pt, max_pt, 21)
print(pt_bins)

ax.hist(bkg_data[:,1], bins=pt_bins, histtype='step', log=True)
ax.hist(w_data[:,1], bins=pt_bins, histtype='step', log=True)
ax.hist(top_data[:,1], bins=pt_bins, histtype='step', log=True)
ax.hist(h80_data[:,1], bins=pt_bins, histtype='step', log=True)

plt.show()
