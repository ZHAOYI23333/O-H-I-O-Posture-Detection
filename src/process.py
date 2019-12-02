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

def sort_index(blocks):
    ind_list = []
    for i in range(4):
        _,_,ci,_,_ = blocks[i]
        for j in range(4):
            if len(ind_list)<=j:
                ind_list.append(blocks[i])
                break
            else:
                _,_,cj,_,_ = ind_list[j]
                if ci < cj:
                    ind_list.insert(j,blocks[i])
                    break
    return ind_list
    
if __name__ == '__main__':


	result_folder = 'result'
	if not os.path.exists(result_folder): os.mkdir(result_folder)
	p1_folder = result_folder + '/p1'
	if not os.path.exists(p1_folder): os.mkdir(p1_folder)
	p2_folder = result_folder + '/p2'
	if not os.path.exists(p2_folder): os.mkdir(p2_folder)
	p3_folder = result_folder + '/p3'
	if not os.path.exists(p3_folder): os.mkdir(p3_folder)
	p4_folder = result_folder + '/p4'
	if not os.path.exists(p4_folder): os.mkdir(p4_folder)
	p_folder = [p1_folder,p2_folder,p3_folder,p4_folder]


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
	config = None
	scaler = 1.5
	diff = [[],[],[],[]]
	old_person = [0]*4
	person = [0]*4
	while success:
	    if not config:
	        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	        print(image.max())
	        blocks = search.search(image_RGB)
	#         print(blocks)
	        blocks = sort_index(blocks)
	#         print(blocks)
	        config = blocks
	        for i in range(4):
	            v,r,c,h,w = config[i]
	            config[i] = (v,
	                         max(0,int(r+(1-scaler)*h/2)),
	                         max(0,int(c+(1-scaler)*w/2)),
	                         int(h*scaler),
	                         int(w*scaler))
	            print(config[i])
	            old_person[i] = image[config[i][1]:config[i][1]+config[i][3],config[i][2]:config[i][2]+config[i][4]]
	#         break
	#     print()
	    else:
	        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	        for i in range(4):
	            person[i] = image[config[i][1]:config[i][1]+config[i][3],config[i][2]:config[i][2]+config[i][4]]
	            Tool.imwrite(person[i],p_folder[i]+'/' + str(count) + '.png')
	#             Tool.imshow(person[i])
	#             Tool.imshow(old_person[i])
	#             diff_img = np.abs(person[i]-old_person[i])
	#             T = Tool.getT(diff_img,t1)
	#             Tool.imshow(diff_img)
	#             print(diff_img.max(),diff_img.min(),T)
	#             diff_img[diff_img<T] = 0
	#             diff_img[diff_img>=T] = 1
	#             Tool.imshow(diff_img)
	#             print(diff_img.max(),diff_img.min(),T)
	            
	            
	#             diff_img = bwmorph(diff_img, option)
	#             L, num = skimage.measure.label(diff_img, 8,return_num=True)
	#             unique, counts = np.unique(L, return_counts=True)
	#             for j in range(len(unique)-1,0,-1):
	#                 if counts[j]<t2:
	#                     counts = np.delete(counts,j)
	#                     unique = np.delete(unique,j)
	#             counts = np.delete(counts,0)
	#             unique = np.delete(unique,0)
	        #     print('unique')
	        #     print(unique)
	        #     print('counts')
	        #     print(counts)
	#             diff_img = np.isin(L, unique.tolist()).astype(np.uint8)
	#             Tool.imshow(diff_img)
	#             break
	#             diff[i].append(diff_img)
	#             Tool.imwrite(diff_img*255,p_folder[i]+'/' + str(count) + '.png')
	#             old_person[i] = person[i]
	    #     Tool.imshow(p1)
	    #     Tool.imshow(p2)
	    #     Tool.imshow(p3)
	    #     Tool.imshow(p4)
	#         persion[0].append(diff_p1)
	#         persion[1].append(diff_p2)
	#         persion[2].append(diff_p3)
	#         persion[3].append(diff_p4)
	#         Tool.imwrite(p1,p1_folder+'/p1_' + str(count) + '.png')
	#         Tool.imwrite(p2,p2_folder+'/p2_' + str(count) + '.png')
	#         Tool.imwrite(p3,p3_folder+'/p3_' + str(count) + '.png')
	#         Tool.imwrite(p4,p4_folder+'/p4_' + str(count) + '.png')
	#     if count == 1: break
	#     Ims.append(image)
	    success, image = videoCapture.read()
	    count += 1
	    print('Read a new frame:', count,end='\r')
	# Ims = np.array(Ims)

