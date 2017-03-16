from utils import *

#Define paths
path = 'data/'
#path = 'data/sample/'
results_path = path + 'results/'
model_path = results_path + 'models/'

batch_size=128

file_path = path+'results/vgg_conv.h5'
f = h5py.File(file_path)
names = HDF5Matrix(file_path, 'test_names').data[:]
features = HDF5Matrix(file_path, 'test_features').data[:]
name2feat = dict(map(lambda tup: (tup[0], tup[1].flatten()), zip(names, features)))
dim = name2feat.values()[0].shape[0]

def pred_ensemble(model_names):
    preds = []
    for model_name in model_names:
        bn_model = load_model(model_path+model_name)
        preds.append(bn_model.predict(features, batch_size=batch_size))
    preds = np.asarray(preds)
    preds = np.mean(preds, axis=0)
    return preds

model_names = ['model.0.0335.hdf5', 'model.0.0337.hdf5', 'model.0.0339.hdf5',
               'model.0.0340.hdf5', 'model.0.0350.hdf5', 'model.0.0362.hdf5',
               'model.0.0400.hdf5'] 
#               'model.0.0539.hdf5', 'model.0.0541.hdf5',
#               'model.0.0542.hdf5', 'model.0.0555.hdf5', 'model.0.0569.hdf5']
preds = pred_ensemble(model_names)
name2pred = dict(zip(names, preds))

def img_maps(names):
    size2name = {}
    for name in names:
        img = Image.open(path+'test/unknown/'+name)
        img_size = str(img.size)
        if(img_size in size2name):
            size2name[img_size].append(name)
        else:
            size2name[img_size] = [name]
    return size2name

def find_knn_preds(names,k):
    features = np.asarray([name2feat[x] for x in names])

    import faiss
    print(len(names))
    index = faiss.IndexFlatL2(dim)
    index.add(features)

    distances, indices = index.search(features, k+1)
    knn_preds = []
    for index in indices:
        knn_preds.append(np.mean(np.asarray([name2pred[names[i]] for i in index]), axis=0))
    test_preds = np.asarray(knn_preds)
    print(test_preds.shape)
    return knn_preds 

def cluster_preds(names, preds):
    init_names = []
    init_preds = []
    size2name = img_maps(names)
    for size,knn_names in size2name.iteritems():
        if(len(knn_names)>60):
            knn_preds = find_knn_preds(knn_names, 5)
        else:
            knn_preds = [name2pred[name] for name in knn_names]
        init_names = init_names + knn_names
        init_preds = init_preds + knn_preds
    return init_names, init_preds

names, preds = cluster_preds(names, preds)

names = np.expand_dims(names, axis=1)
preds = do_clip(preds, 0.95)
submission = np.hstack((names, preds))
np.savetxt('submit1.csv', submission, delimiter=',', fmt='%s', 
           header='image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT', comments='')
