import struct
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
g = open('combined_feature_NUS_Train.pickle','rb')
features = pickle.load(g)
g.close()
nbrs = NearestNeighbors(n_neighbors=500, algorithm='auto').fit(features)
