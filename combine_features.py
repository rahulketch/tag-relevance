import numpy as np
import scipy.sparse as sps
import pickle
"""Early Fusion for Features"""
#path = "../mirflickr/meta/tags/"
path = "../NUS-WIDE-Lite/NUS-WIDE-Lite_features/"
ch = np.loadtxt(path+'Normalized_CH_Lite_Train.dat')

cm = np.loadtxt(path+'Normalized_CM55_Lite_Train.dat')
corr = np.loadtxt(path+'Normalized_CORR_Lite_Train.dat')
edh = np.loadtxt(path+'Normalized_EDH_Lite_Train.dat')
wt = np.loadtxt(path+'Normalized_WT_Lite_Train.dat')
combined = np.hstack((ch,cm,corr,edh,wt))
g = open('combined_feature_NUS_Train.pickle','wb')
pickle.dump(combined,g)
g.close()

