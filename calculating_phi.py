import pickle
import numpy as np
k = 10
path_nus = '/home/kiran/Rahul/NUS-WIDE-Lite/'
from sklearn.neighbors import NearestNeighbors
g = open('combined_feature_NUS_Train.pickle','rb')
features = pickle.load(g)
g.close()
#nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features)
s = len(features)
mu = 4*s
temp = np.transpose(np.loadtxt(path_nus+'NUS-WIDE-Lite_tags/Lite_Tags1k_Train.txt'))
print(len(temp),len(temp[0]))
print(temp[0])