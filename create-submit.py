from utils import *

#Define paths
path = 'data/'
#path = 'data/sample/'
results_path = path + 'results/'
model_path = results_path + 'models/'

batch_size=128

file_path = path+'results/vgg_conv.h5'
f = h5py.File(file_path)
test_names = HDF5Matrix(file_path, 'test_names').data[:]
#test_names = [test_name.split('.')[0] for test_name in test_names]
conv_test_feat = HDF5Matrix(file_path, 'test_features')

def pred_ensemble(model_names):
    preds = []
    for model_name in model_names:
        bn_model = load_model(model_path+model_name)
        preds.append(bn_model.predict(conv_test_feat, batch_size=batch_size))
    preds = np.asarray(preds)
    preds = np.mean(preds, axis=0)
    return preds

model_names = ['model.0.0335.hdf5', 'model.0.0337.hdf5', 'model.0.0339.hdf5',
               'model.0.0340.hdf5', 'model.0.0350.hdf5', 'model.0.0362.hdf5',
               'model.0.0400.hdf5', 'model.0.0539.hdf5', 'model.0.0541.hdf5',
               'model.0.0542.hdf5', 'model.0.0555.hdf5', 'model.0.0569.hdf5']
preds = pred_ensemble(model_names)
names = np.expand_dims(test_names, axis=1)
preds = do_clip(preds, 0.9)

submission = np.hstack((names, preds))
np.savetxt('submit1.csv', submission, delimiter=',', fmt='%s', 
           header='image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT', comments='')
