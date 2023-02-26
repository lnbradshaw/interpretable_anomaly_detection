import os
import requests
import numpy as np
import h5py
import energyflow as ef
from time import perf_counter
from tqdm import tqdm

def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    dlen = int(r.headers.get('Content-Length', '0'))
    if r.ok:
        print("saving to", os.path.relpath(file_path))
        progress_bar = tqdm(total=dlen,unit='B',unit_scale=True,colour="green")
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

def center_rotate_reflect(data):
    """
    data: array with 4 momenta for each jet constituent.
        Expected shape is [num_events, num_constituents, 4] since we only need
        pt, eta, phi, m (from px, py, pz, m)

    """
    # start by converting (px, py, pz) to (pt, eta, phi), then rotate and reflect so
    # pt weighted centroid of each jet is in the upper right hand quadrant of the
    # eta-phi plane
    data = np.asarray([ef.utils.reflect_ptyphims(ef.utils.rotate_ptyphims(ef.utils.ptyphims_from_p4s(x), rotate='ptscheme', center='ptscheme')) for x in data])
    # make sure all of the phi values for each constituent are in a 2pi range
    data[:,:,2] = ef.utils.phi_fix(data[:,:,2], phi_ref=data[np.argmax(data[:,0,0]),0,2])

    return data

def make_image(data, npix: int=20, img_width: float=2):
    """
    data: array with shape [num_events, num_constituents, 4]. Images need to be
        made with output from center_rotate_reflect().
    """
    image = np.asarray([ef.utils.pixelate(x, npix=npix, img_width=img_width, nb_chan=1, norm=False, charged_counts_only=False) for x in data])
    # doing pt normalization separately. the pixelate() function has this built in,
    # but I had issues getting it to work.
    for img in image:
        normfactor = np.sum(img[...,0])
        if normfactor != 0:
            img /= normfactor

    return image

def apply_pt_cut(data, pt, pt_min: float =550, pt_max: float =650):
    """
    data: array of data to apply pt cut to

    pt: array of pts corresponding to same file as data
    """
    pt = pt.reshape(-1,)
    data_int = data[pt>=pt_min]
    data_final = data_int[pt[pt>=pt_min]<=pt_max]

    return data_final

DATADIR ='../data/'

def main():
    start = perf_counter()
    dir_names = ['raw/','processed/efps/','processed/images/','processed/p4s/','processed/nsub/','processed/train_test_data/']
    if not os.path.exists(DATADIR):
        os.makedirs(DATADIR)
    for name in dir_names:
        if not os.path.exists(DATADIR+name):
            os.makedirs(DATADIR+name)
    # start by downloading all of the files from Zenodo
    # starting with the signals
    sig_url = 'https://zenodo.org/record/4614656/files/'
    fnames = ['top_100k_01.h5','W_100k_01.h5','h3_m174_h80_100k_01.h5']
    for name in fnames:
        if not os.path.exists(DATADIR+'raw/'+name):
            print(f"Downloading {name} from {sig_url}")
            download(url=sig_url+name, dest_folder=DATADIR+'raw/')
            print('')
    # and now the background
    bname = 'qcd_dijet.h5'
    bkg_url = 'https://zenodo.org/record/4641460/files/'
    if not os.path.exists(DATADIR+'raw/'+bname):
        print(f"Downloading {bname} from {bkg_url}")
        download(url=bkg_url+bname, dest_folder=DATADIR+'raw/')
        print('')

    # now, let's preprocess all of the data we just downloaded
    for filename in os.listdir(DATADIR+'raw/'):
        print('Preprocessing raw data at '+str(os.path.join(DATADIR+'raw/', filename)))
        # loading in the raw data file
        with h5py.File(os.path.join(DATADIR+'raw/', filename), 'r') as hf:
            raw_data = hf['objects/jets/constituents'][:,0]
        # finding the maximum number of constiuents in any event
        num_const = int(max(raw_data[i].shape[0] for i in range(raw_data.shape[0]))/5)
        # zero pad each event until they all have num_const number of constiuents
        data = np.zeros((raw_data.shape[0], num_const, 5))
        fill_arr = np.array([0., 0., 0., 0., 0.,])
        for i,d in enumerate(raw_data):
            if d.shape[0] >= num_const*5:
                data[i, ...] = d[:num_const*5].reshape(num_const,5)
            else:
                while d.shape[0] < num_const*5:
                    d = np.hstack((d, fill_arr))
                data[i, ...] = d.reshape(num_const, 5)
        p4s_arr = data[:,:,:4]
        # save processed 4-momenta (NOTE: pt cut is NOT applied to this data)
        print(f"Preprocessed and saving 4-momenta for {filename.rsplit('.',1)[0]}, making images")
        with h5py.File(os.path.join(DATADIR+'processed/p4s', filename.rsplit('.',1)[0]+'_p4s.h5'), 'w') as hf:
            hf.create_dataset("p4s", data=p4s_arr)
        # compute pts for file
        pts= np.asarray([ef.pts_from_p4s(np.sum(x, axis=0)) for x in p4s_arr])

        # load in n-subjettiness observables
        with h5py.File(os.path.join(DATADIR+'raw', filename), 'r') as hf:
            nsub = hf['objects/jets/obs'][:,0]
        #print(nsub.shape)
        nsub = apply_pt_cut(nsub[:,4:],pts)
        #print(nsub.shape)
        print(f"Saving nsubjettiness for {filename.rsplit('.',1)[0]}")
        with h5py.File(os.path.join(DATADIR+'processed/nsub/', filename.rsplit('.',1)[0]+'_nsub.h5'), 'w') as hf:
            hf.create_dataset("nsub", data=nsub)

        p4s_arr = apply_pt_cut(p4s_arr, pts)

        # create and save jet images
        image = make_image(center_rotate_reflect(p4s_arr), npix=40, img_width=2)
        print(f"Saving image for {filename.rsplit('.',1)[0]}, computing EFPs")
        with h5py.File(os.path.join(DATADIR+'processed/images/', filename.rsplit('.',1)[0]+'20x20_img.h5'), 'w') as hf:
            hf.create_dataset("img", data=image)

        # compute and save efps from 4-momenta
        dmax = 5
        beta = 1
        kappa = 1
        print(f"Calculating d <= {dmax} EFPs for {p4s_arr.shape[0]} jets from {filename.rsplit('.',1)[0]}... ", end='')
        efpset = ef.EFPSet(('d<=', dmax), measure='hadr', beta=beta, kappa=kappa, coords='epxpypz')
        masked = [x[x[:,0]>0] for x in p4s_arr]
        efps = efpset.batch_compute(masked)
        print('Done')

        with h5py.File(os.path.join(DATADIR+'processed/efps/', filename.rsplit('.',1)[0]+'_d5k1b1.h5'), 'w') as hf:
            hf.create_dataset("efps", data=efps)

        # separately compute masses
        masses = np.abs(np.asarray([ef.ms_from_p4s(np.sum(x,axis=0)) for x in p4s_arr]))
        masses = masses.reshape(-1,1)

        pt_min = 550
        pt_max = 650
        pts = pts[pts>=pt_min]
        pts = pts[pts<=pt_max]
        pts = pts.reshape(-1,1)
        # quick check to make sure pt cut was applied properly
        #print(min(pts[:,0]), max(pts[:,0]))

        # combine masses, pts, efps into single array for training and testing
        train_test_data = np.hstack((masses, pts, efps[:,1:]))
        print(f"Saving train/test data for {filename.rsplit('.',1)[0]}")
        with h5py.File(os.path.join(DATADIR+'processed/train_test_data/', filename.rsplit('.',1)[0]+'_train_test_data.h5'), 'w') as hf:
            hf.create_dataset("train_test_data", data=train_test_data)
        print('')
    end = perf_counter()
    print(f'This script took {np.round(end-start,3)} seconds to run.')

if __name__ == '__main__':
    main()
