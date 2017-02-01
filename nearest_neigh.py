"""Nearest Neighbour code"""
import struct
import numpy as np
from sklearn.neighbors import NearestNeighbors
with open("../vgg-verydeep-16-fc7relu/feature.bin", mode='rb') as file: # b is important -> binary
	fileContent = file.read()
f = open('../vgg-verydeep-16-fc7relu/id.txt',mode='r')
list_image = f.read().split()
#print(list_image)
print(len(list_image))
f.close()
x = struct.unpack("f" * ((len(fileContent)) // 4), fileContent)
print(len(x))
x_ = np.asarray(x)
x_res = np.resize(x_,(25000,4096))
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(x_res)
print('stuck')
#distances, indices = nbrs.kneighbors(x_res)


	
