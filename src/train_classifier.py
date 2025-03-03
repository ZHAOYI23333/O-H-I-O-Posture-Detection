import os
import sys
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
import skimage.transform as tf
import skimage.color as clr
from skimage.feature import hog
from sklearn import svm, tree, neural_network

data_path = '../data/'
train_pos_list = '../data/train/pos.lst'
train_neg_list = '../data/train/neg.lst'
test_pos_list = '../data/test/pos.lst'
test_neg_list = '../data/test/neg.lst'
classifier_path = '../save/classifier.sav'
img_size = (128, 64)
negative_crop_count = 10
downsampling_filter = lambda img:ndimg.gaussian_filter(img, sigma = 1, mode = 'nearest', truncate = 3)

classifier_constructor = lambda :svm.SVC(kernel = 'linear', probability = True) # 0.9741
# classifier_constructor = lambda :tree.DecisionTreeClassifier() # 0.8625
# classifier_constructor = lambda :neural_network.MLPClassifier(hidden_layer_sizes = (512, ), activation = 'relu', solver = 'sgd', batch_size = 32, learning_rate = 'constant', learning_rate_init = 0.001, max_iter = 20) # 0.9767

window_size = lambda :img_size # for external access

def print_flush(s):
    print(s)
    sys.stdout.flush()

hog_descriptor = lambda img:hog(img, orientations = 9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm = 'L2', multichannel = True, transform_sqrt = True, feature_vector = True)
    
def format_image(img):
    if img.shape[-1] == 4:
        img = clr.rgba2rgb(img)
    elif img.shape[-1] != 3:
        raise Exception('Only taking RGB or RGBA images')
        
    if img.shape[0] != img_size[0] or img_shape[1] != img_size[1]:
        if img.shape[0] < img_size[0] or img.shape[0] - img_size[0] != img.shape[1] - img_size[1]: # not padding
            scale = max(img_size[0] / img.shape[0], img_size[1] / img.shape[1])
            if scale < 1:
                img = downsampling_filter(img)
            img = tf.resize(img, (round(img.shape[0] * scale), round(img.shape[1] * scale)))
        start = ((img.shape[0] - img_size[0]) // 2, (img.shape[1] - img_size[1]) // 2)
        img = img[start[0]:start[0] + img_size[0], start[1]:start[1] + img_size[1]]
    if np.max(img) > 1:
        img = img / 255 # normalize to 0 - 1
    return img
    
def random_crop(img, window_size, crop_number):
    if crop_number:
        if not window_size:
            raise Exception('Window size has to be assigned to a value when random crop is enabled')
        else:
            try:
                window_size[0]
                if len(window_size) != 2:
                    raise Exception('Window size should be two dimensional')
            except:
                raise Exception('Window size should be an array-like object')
                
    if window_size[0] > img.shape[0] or window_size[1] > img.shape[1]:
        raise Exception('Window size should be smaller than image size')
        
    row, col = random.randint(0, img.shape[0] - window_size[0]), random.randint(0, img.shape[1] - window_size[1])
    return [img[row:row + window_size[0], col:col + window_size[1]] for i in range(crop_number)]
    
def read_images(path_list, crop_number = 0, window_size = None):
    print_flush('Loading %s ...' % path_list)
    paths = []
    with open(path_list, 'r') as lst:
        paths = lst.read().split("\n")
    while len(paths) > 0 and paths[-1] == '':
        paths.pop(-1)
    img_list = [plt.imread(os.path.join(data_path, path), 'jpg') for path in paths]
    ret_list = [format_image(img) for img in img_list]
    if crop_number:
        for img in img_list:
            ret_list.extend(random_crop(img, window_size, crop_number))
    return ret_list
    
if __name__ == '__main__':
    train_pos_images = read_images(train_pos_list)
    train_neg_images = read_images(train_neg_list, negative_crop_count, img_size)
    train_images = train_pos_images + train_neg_images
    train_labels = [1 for i in range(len(train_pos_images))] + [0 for i in range(len(train_neg_images))]

    print_flush('Generating HoG descriptors for train images...')
    train_hog_descriptors = [hog_descriptor(img) for img in train_images]

    image_classifier = classifier_constructor()

    print_flush('Fitting...')
    image_classifier.fit(train_hog_descriptors, train_labels)

    test_pos_images = read_images(test_pos_list)
    test_neg_images = read_images(test_neg_list)
    test_images = test_pos_images + test_neg_images
    test_labels = [1 for i in range(len(test_pos_images))] + [0 for i in range(len(test_neg_images))]

    print_flush('Generating HoG descriptors for test images...')
    test_hog_descriptors = [hog_descriptor(img) for img in test_images]

    print_flush('Evaluating...')
    accuracy = image_classifier.score(test_hog_descriptors, test_labels)
    print_flush('SVM accuracy: %f' % accuracy)

    pickle.dump(image_classifier, open(classifier_path, 'wb+'))
