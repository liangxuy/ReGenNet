import os
import pickle
import h5py
import numpy as np

SRC_H5 = 'dataset/ntu120/smplx/ntu_2p_smplx.h5'
DEST_H5 = 'dataset/ntu120/smplx/conditioned/ntu_2p_smplx_cond.h5'
label_folder = 'dataset/ntu120/smplx/Results'

f_out = h5py.File(DEST_H5, 'w')

with h5py.File(SRC_H5, 'r') as f:
    sample_name = list(f.keys())
    for i, filename in enumerate(sample_name):
        label_file = os.path.join(label_folder, filename+'.mp4.txt')
        label = int(open(label_file, 'r').readlines()[0])
        if label == 0:
            tmp = f[filename]
        elif label == 1:
            tmp = np.zeros_like(f[filename])
            tmp[:,:,0:3] = f[filename][:,:,3:6]
            tmp[:,:,3:6] = f[filename][:,:,0:3]
        f_out.create_dataset(filename, data=tmp, dtype='f')
f_out.close()
