#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import numpy.matlib as matlib
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import skimage.feature
import scipy


# In[16]:


def imread(file_name, dtype=float):
    # Read in image file
    # Return type: numpy.ndarray
    # [0,255] uint8
    return np.array(Image.open(file_name))


# In[17]:


def imshow(image, cmap=None, title=None):
    # Show the input image
    # Set cmap='gray' if showing gray scale images
    # title: the title of image
    if len(image.shape) == 2:
        cmap = 'gray'
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=cmap)


# In[31]:


def imwrite(image, file_name, norm=None):
    # gray scale must have pixel value in [0,255], may change with rgb
#     image = image*255
    if norm:
        image = normalize(image)
    image = image.astype(np.uint8)
    Image.fromarray(image.astype(np.uint8)).save(file_name)


# In[20]:


def rgb2gray(image):
    return skimage.color.rgb2gray(image)


# In[21]:


def repmat(image, r, c):
    return matlib.repmat(image, r, c)


# ### Homework 2

# In[22]:


def normalize(img, normType=1):
    # Normalize the input image
    # Transform its range to [0, 255]
#     img = img.astype(np.uint8)
    if normType == 1:
        img = img - img.min()
        return img*255/img.max()
    elif normType == 2:
        img = img - img.min()
        return img + img.mean()


# In[23]:


def fspecial(ftype, hsize=None, sigma=None):
    if ftype == 'gaussian':
        s = (hsize-1.)/2
        y, x = np.ogrid[-s:s+1, -s:s+1]
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[ h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    elif ftype == 'sobel':
        return np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])
    else:
        print("Please use other filter types")
        exit()


# In[24]:


def imfilter(image, mask, mode='nearest'):
    return scipy.ndimage.convolve(image, mask, mode=mode)


# In[25]:


def gaussDeriv2D(sigma):
    s = np.ceil(3*sigma)
    y, x = np.ogrid[-s:s+1, -s:s+1]
    Gx= -x/(2*np.pi*np.power(sigma, 4))*np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    Gy= -y/(2*np.pi*np.power(sigma, 4))*np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    return Gx, Gy


# In[26]:


def getT(img, T, mode='percentile'):
    if mode == 'percentile':
        return np.percentile(img, T)
    else:
        return (img.max()-img.min())*T+img.min()


# In[28]:


def apply_T(img, T):
    image = img.copy()
    image[image <= T] = 0
    image[image > T] = 1
    return image


# In[29]:


def canny(image, sigma=2, low_threshold=0.7, high_threshold=0.8, use_quantiles=True):
    return skimage.feature.canny(Im, sigma, low_threshold, high_threshold, use_quantiles)


# In[ ]:




