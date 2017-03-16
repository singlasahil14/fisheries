from os import listdir
import numpy as np

data_path ='data/'

# Get filenames and class ids
fnames = listdir(data_path+'test/')
num_rows = len(fnames)
fnames = np.expand_dims(np.asarray(fnames), 1)
print(fnames.shape)
probs = np.tile([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], (num_rows,1))
print(probs.shape)
to_submit = np.hstack((fnames, probs))
np.savetxt('submit.csv', to_submit, fmt='%s', delimiter=',', 
           header='image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT', comments='')
