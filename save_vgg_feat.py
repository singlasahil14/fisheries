from utils import *

#Define path
path = "data/"
#path = "data/sample/"
results_path = path + 'results/'

input_shape = (224,224)

#Define conv model
from vgg16bn import Vgg16BN
vgg16 = Vgg16BN()
vgg16.ft(8)
conv_layers,fc_layers = split_at(vgg16.model, Convolution2D)
conv_model = Sequential(conv_layers)

batch_size = 64
gen = image.ImageDataGenerator()
train_datagen = gen.flow_from_directory(path+'train/', target_size=input_shape,
                                        batch_size=batch_size)
test_datagen = gen.flow_from_directory(path+'test/', target_size=input_shape, 
                                       batch_size=batch_size, shuffle=False)

s = FeatureSaver(train_datagen=train_datagen, 
                 test_datagen=test_datagen)
f = h5py.File(results_path+'vgg_conv.h5', 'w')
s.save_train(conv_model, f, num_epochs=1)
s.save_test(conv_model, f)

shutil.copy(results_path+'vgg_conv.h5', results_path+'vgg_conv_cp.h5')

print(f['train_features'].shape)
print(f['train_labels'].shape)

print(f['test_features'].shape)
print(f['test_names'].shape)
