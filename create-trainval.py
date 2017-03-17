from utils import *

def size_map(img_dir='data/train/'):
    size2names = defaultdict(list)
    for img_path in glob(img_dir+'/*/*.jpg'):
        img = Image.open(img_path)
        img_size = img.size
        img_name = img_path.split('/')[-1]
        class_name = img_path.split('/')[-2]
        size2names[img_size].append((img_name, class_name))
    return size2names

LAG_vfnames = [ 'img_04481.jpg', 'img_05449.jpg', 'img_06876.jpg', 'img_02415.jpg', 'img_05123.jpg', 
                'img_03940.jpg', 'img_01644.jpg', 'img_01952.jpg', 'img_07830.jpg', 'img_04711.jpg', 
                'img_05939.jpg', 'img_02349.jpg', 'img_06129.jpg', 'img_03774.jpg', 'img_00176.jpg',
                'img_02728.jpg', 'img_04435.jpg', 'img_06257.jpg', 'img_02186.jpg', 'img_04352.jpg', 
                'img_03232.jpg', 'img_04958.jpg', 'img_02236.jpg', 'img_07334.jpg', 'img_01527.jpg',
                'img_00657.jpg', 'img_01037.jpg']

path = 'data/'
size2names = size_map(path+'train')
all_list = []
for k,v in size2names.items():
    all_list = all_list + v
NoF_vfnames = [tup for tup in size2names[(1280, 750)] if tup[1]=='NoF']
OTHER_vfnames = [tup for tup in size2names[(1280, 750)] if tup[1]=='OTHER']
val_list = [(img_name, 'LAG') for img_name in LAG_vfnames] + size2names[(1280, 974)] + NoF_vfnames
val_list = val_list + OTHER_vfnames
train_list = np.asarray([tup for tup in all_list if tup not in val_list])
val_list = np.asarray(val_list)

def validate(im_list):
    class2len = defaultdict(int)
    size2classes = defaultdict(lambda: defaultdict(int))
    for im_name,class_name in im_list:
        img_path = 'data/train/' + class_name + '/' + im_name
        img = Image.open(img_path)
        img_size = img.size
        size2classes[img_size][class_name] = size2classes[img_size][class_name] + 1
        class2len[class_name] = class2len[class_name] + 1
    return size2classes, class2len

def print_dict(dic):
    for k,v in dic.items():
        print('{}: {}'.format(k,v))

print(len(train_list))
print(len(val_list))

np.savetxt('train_list.csv', train_list, delimiter=',', fmt='%s', 
           header='img_name, class_name', comments='') 
np.savetxt('val_list.csv', val_list, delimiter=',', fmt='%s', 
           header='img_name, class_name', comments='') 

size2classes, class2len = validate(train_list)
print('Printing for training')
print_dict(size2classes)
print_dict(class2len)

size2classes, class2len = validate(val_list)
print('Printing for validation')
print_dict(size2classes)
print_dict(class2len)
