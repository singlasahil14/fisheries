from utils import *

#Define paths
path = 'data/'
#path = 'data/sample/'
results_path = path + 'results/'
model_path = results_path + 'models/'
if not os.path.exists(model_path): os.makedirs(model_path)

#Define model
def get_bn_layers(p, num=128):
    return [
        MaxPooling2D(input_shape=(512,14,14)),
        Flatten(),
        Dense(num, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(num, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(8, activation='sigmoid')
        ]

file_path = path+'results/vgg_conv.h5'
f = h5py.File(file_path)
features = f['train_features'][:]
labels = f['train_labels'][:]
from sklearn.model_selection import train_test_split
conv_trn_feat, conv_val_feat, trn_labels, val_labels = train_test_split(features, labels, test_size=500, stratify=labels)
del features
del labels

batch_size=64
num_samples = len(conv_trn_feat)
trn_datagen = gen_batches(conv_trn_feat, trn_labels, batch_size, 
                          epoch_size=num_samples)
checkpointer = ModelCheckpoint(filepath=model_path+'model.{val_loss:.4f}.hdf5', 
                               verbose=0, save_best_only=True)

p = 0.5
num_hidden = 4096
bn_model = Sequential(get_bn_layers(p, num=num_hidden))
bn_model.compile(Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
bn_model.fit_generator(trn_datagen, samples_per_epoch=num_samples, nb_epoch=4, 
             validation_data=(conv_val_feat, val_labels), callbacks=[checkpointer], 
             max_q_size=100)
bn_model.optimizer.lr.set_value(3e-5)
bn_model.fit_generator(trn_datagen, samples_per_epoch=num_samples, nb_epoch=4, 
             validation_data=(conv_val_feat, val_labels), callbacks=[checkpointer], 
             max_q_size=100)
bn_model.optimizer.lr.set_value(3e-6)
bn_model.fit_generator(trn_datagen, samples_per_epoch=num_samples, nb_epoch=4, 
             validation_data=(conv_val_feat, val_labels), callbacks=[checkpointer], 
             max_q_size=100)
