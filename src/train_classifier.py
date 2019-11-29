import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
import skimage.transform as tf
import skimage.color as clr
from skimage.feature import hog
from sklearn import svm

data_path = '../data/'
train_pos_list = '../data/train/pos.lst'
train_neg_list = '../data/train/neg.lst'
test_pos_list = '../data/test/pos.lst'
test_neg_list = '../data/test/neg.lst'
classifier_path = '../save/classifier.sav'
img_size = (60, 36)
window_size = lambda :img_size # for external access
classifier_constructor = lambda :svm.SVC(probability = True)
downsampling_filter = lambda img:ndimg.gaussian_filter(img, sigma = 1, mode = 'nearest', truncate = 3)

def print_flush(s):
    print(s)
    sys.stdout.flush()

hog_descriptor = lambda img:hog(img, orientations = 9, pixels_per_cell=(6, 6), cells_per_block=(3, 3), block_norm = 'L2', multichannel = True, feature_vector = True)
    
def format_image(img):
    if img.shape[-1] == 4:
        img = clr.rgba2rgb(img)
    elif img.shape[-1] != 3:
        raise Exception('Only taking RGB or RGBA images')
        
    if img.shape[0] != img_size[0] and img.shape[1] != img_size[1]:
        scale = max(img_size[0] / img.shape[0], img_size[1] / img.shape[1])
        if scale < 1:
            img = downsampling_filter(img)
        img = tf.resize(img, (round(img.shape[0] * scale), round(img.shape[1] * scale)))
        start = ((img.shape[0] - img_size[0]) // 2, (img.shape[1] - img_size[1]) // 2)
        img = img[start[0]:start[0] + img_size[0], start[1]:start[1] + img_size[1]]
    return img
    
def read_images(path_list):
    print_flush('Loading %s ...' % path_list)
    paths = []
    with open(path_list, 'r') as lst:
        paths = lst.read().split("\n")
    while len(paths) > 0 and paths[-1] == '':
        paths.pop(-1)
    img_list = [format_image(plt.imread("%s%s" % (data_path, path), 'jpg')) for path in paths]
    return img_list
    
if __name__ == '__main__':
    train_pos_images = read_images(train_pos_list)
    train_neg_images = read_images(train_neg_list)
    train_images = train_pos_images + train_neg_images
    train_labels = [1 for i in range(len(train_pos_images))] + [0 for i in range(len(train_neg_images))]

    print_flush('Generating HoG descriptors for train images...')
    train_hog_descriptors = [generate_hog(img) for img in train_images]

    image_classifier = classifier_constructor()

    print_flush('Fitting...')
    image_classifier.fit(train_hog_descriptors, train_labels)

    test_pos_images = read_images(test_pos_list)
    test_neg_images = read_images(test_neg_list)
    test_images = test_pos_images + test_neg_images
    test_labels = [1 for i in range(len(test_pos_images))] + [0 for i in range(len(test_neg_images))]

    print_flush('Generating HoG descriptors for test images...')
    test_hog_descriptors = [generate_hog(img) for img in test_images]

    print_flush('Evaluating...')
    accuracy = image_classifier.score(test_hog_descriptors, test_labels)
    print_flush('SVM accuracy: %f' % accuracy)

    pickle.dump(image_classifier, open(classifier_path, 'wb+'))
