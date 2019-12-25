import pickle

import scipy.ndimage as ndimg
import skimage.transform as tf
import matplotlib.pyplot as plt

from train_classifier import window_size, hog_descriptor

image_path = '../data/search_test.png'
classifier_path = '../save/nn_svm_classifier.sav'
window_size = window_size()
step = (6, 8, 10)
scale = (0.2, 0.25, 0.3) # 1280 x 720
iou_threshold = 0.25
classifier = pickle.load(open(classifier_path, 'rb'))
downsampling_filter = lambda img:ndimg.gaussian_filter(img, sigma = 1, mode = 'nearest', truncate = 3)

def intersection_over_union(loc1, loc2):
    def len2coord(loc):
        another_row_coord = loc[0] + loc[2]
        another_col_coord = loc[1] + loc[3]
        loc = (min(loc[0], another_row_coord), min(loc[1], another_col_coord), max(loc[0], another_row_coord), max(loc[1], another_col_coord))
        return loc
        
    area_sum = loc1[2] * loc1[3] + loc2[2] * loc2[3]
    
    loc1 = len2coord(loc1)
    loc2 = len2coord(loc2)
    if loc1[2] <= loc2[0] or loc2[2] <= loc1[0] or loc1[3] <= loc2[1] or loc2[3] <= loc1[1]:
        return 0
    intersection_row = max(loc1[2], loc2[2]) - min(loc1[0], loc2[0]) - abs(loc1[0] - loc2[0]) - abs(loc1[2] - loc2[2])
    intersection_col = max(loc1[3], loc2[3]) - min(loc1[1], loc2[1]) - abs(loc1[1] - loc2[1]) - abs(loc1[3] - loc2[3])
    intersection_area = intersection_row * intersection_col
    union_area = area_sum - intersection_area
    return intersection_area / union_area

def non_maximum_suppression(block_list):
    block_list.sort(reverse = True)
    index = 0
    while index < len(block_list):
        match_index = index + 1
        while match_index < len(block_list):
            iou = intersection_over_union(block_list[index][1:], block_list[match_index][1:])
            if iou > iou_threshold:
                block_list.pop(match_index)
            else:
                match_index += 1
        index += 1

def search(img, window_size = window_size, scale = scale, step = step, classifier = classifier):
    try:
        scale[0]
        try:
            step[0]
            if len(step) != len(scale):
                raise Exception('The size of step must match the size of scale')
        except:
            raise Exception('scale is an array-like object but step is not')
    except:
        try:
            step[0]
            raise Exception('step is an array-like object but scale is not')
        except:
            scale = (scale, )
            step = (step, )
    if len(window_size) != 2:
        raise Exception('window size shoule be of length 2')
        
    valid_blocks = []
    for index in range(len(scale)):
        if scale[index] < 1:
            current_img = downsampling_filter(img)
        current_img = tf.rescale(current_img, scale[index])
        for i in range(0, current_img.shape[0] - window_size[0], step[index]):
            for j in range(0, current_img.shape[1] - window_size[1], step[index]):
                current_hog_descriptor = hog_descriptor(current_img[i:i + window_size[0], j:j + window_size[1]])
                prob = classifier.predict_proba([current_hog_descriptor])[0]
                if prob[1] > prob[0]:
                    valid_blocks.append((prob[1], round(i / scale[index]), round(j / scale[index]), round(window_size[0] / scale[index]), round(window_size[1] / scale[index])), )
                    
    non_maximum_suppression(valid_blocks)
    return valid_blocks
    
if __name__ == '__main__':
    img = plt.imread('%s' % image_path, 'jpg')
    blocks = search(img)
    plt.imshow(img)
    for block in blocks:
        plt.gca().add_patch(plt.Rectangle((block[2], block[1]), block[4], block[3], fill = False, edgecolor = 'r', linewidth = 2))
    plt.show()