# Interpretable Anomaly Detection

This repository contains all of the code needed to replicate the results in [Creating Simple, Interpretable Anomaly Detectors for New Physics in Jet Substructure](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.035014). If either this code or its results prove useful to your work, please cite the following
```bibtex
@article{Bradshaw:2022qev,
    author = "Bradshaw, Layne and Chang, Spencer and Ostdiek, Bryan",
    title = "{Creating simple, interpretable anomaly detectors for new physics in jet substructure}",
    eprint = "2203.01343",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/PhysRevD.106.035014",
    journal = "Phys. Rev. D",
    volume = "106",
    number = "3",
    pages = "035014",
    year = "2022"
}
```

## Dependencies

This code requires the following packages to run
- [NumPy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)
- [h5py](https://www.h5py.org)
- [TensorFlow](https://www.tensorflow.org)
- [tqdm](https://tqdm.github.io)
- [scikit-learn](https://scikit-learn.org/stable/)
- [EnergyFlow](https://energyflow.network/installation/)

## Getting Started
After installing the necessary dependencies, start by running ```code/preprocess_data.py```. This script will download all of the [background](https://zenodo.org/record/4641460#.ZCIkuC-B1hE) and [signal](https://zenodo.org/record/4614656#.ZCIkwS-B1hE) samples, and save them to a new directory ```data/raw/[qcd_djiet|top_100k_01|W_100k_01|h3_m174_h80_100k_01|etc.].h5```. It will also compute the particle 4-momenta, Nsubjettiness observables, images, and Energy Flow Polnomials (EFPs), and save them to ```data/processed/p4s/filename.h5```, ```data/processed/nsub/filename.h5```, ```data/processed/images/filename.h5```, and ```data/processed/efps/filename.h5```, respectively. The EFPs are combined with the particle masses and transverse momenta, and are then separately saved to ```data/processed/train_test_data/filename.h5```. 

If you wish to change the transverser momentum cut to different range than the paper ($p_{T}\in\[550,\ 650\]$ GeV), please change the ```pt_min``` and ```pt_max``` values (lines 162 and 163). 

After running ```code/preprocess_data.py```, you should then run ```code/scale_train_test_data.py```. This script will load all of the files in ```data/processed/train_test_data```, use the first 2/3 of the background sample to fit the ```StandardScaler``` from ```scikit-learn```, and then apply this transformation to all of the other data. The resulting files are then saved to ```data/processed/train_test_data/preprocessed_and_scaled/filename.h5```. These preprocessed and scaled data are then used to train the mimicker networks.

Once all of the data has been downloaded and preprocessed, you should then run ```code/train_ae.py```, which trains an image-based autoencoder used as both a baseline anomaly detector, and the model we use to iteratively train our mimcker networks. These mimicker networks can be iteratively trained by running ```code/train_hlns.py``` and ```code/train_pnns.py```, which train the High-Level Network mimickers and Paired Neural Network mimckers, resepectively. For details on how these neural networks work, see Sec. III B 3 and III B 4 of our paper.  

##

###### If you have any questions or concerns, please contact [Layne Bradshaw](mailto:layne.bradsh@gmail.com) 
