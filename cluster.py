from utils import *

#Define paths
path = 'data/'
#path = 'data/sample/'
results_path = path + 'results/'
model_path = results_path + 'models/'
if not os.path.exists(model_path): os.makedirs(model_path)

files_map = {}
sizes = []
for img_file in glob(path+'train/*/'+'*.jpg'):
    img = Image.open(img_file)
    img = img.resize( [int(0.125 * s) for s in img.size] )
    img_size = str(img.size)
    if(img_size in file_sizes)
        files_map[img_size].append(img_file)
    else:
        files_map = [img_file]
    sizes.append(img_size)

#from collections import Counter
#keys, values = zip(*Counter(sizes).items())
#li = zip(keys, values)
#li = sorted(li, key=lambda x: x[1])
#print(li)

def sizewise_cluster(files_map):
    
