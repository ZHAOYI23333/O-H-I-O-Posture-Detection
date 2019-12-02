#import packages
import Tool
import numpy as np
import numpy.matlib
from numpy import linalg as LA
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import skimage.feature
import scipy
from scipy import linalg as SLA
import cv2
import os
import time
import search


if __name__ == '__main__':
	videoCapture = cv2.VideoCapture('../data/video/OHIO/1.mp4')
	fps = videoCapture.get(cv2.CAP_PROP_FPS)
	size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
	        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	print(fps)
	print(type(videoCapture))
	success = True
	count = 0
	Ims = []
	success, image = videoCapture.read()
	while success:
	#     image = Tool.rgb2gray(image)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		Ims.append(image)
		success, image = videoCapture.read()
		count += 1
		print('Read a new frame:', count,end='\r')
	Ims = np.array(Ims)

	blocks = search.search(Ims[0])[:4]

	fig,ax = plt.subplots(1)
	ax.imshow(Ims[0])
	for element in blocks:
	    _,r,c,height,width = element
	    rect = patches.Rectangle((c,r),width,height,linewidth=1,edgecolor='b',facecolor='none')
	    ax.add_patch(rect)
	plt.title('Best_match')
	plt.savefig('Best_match.png',dpi=300)
	plt.show()