import pickle
import numpy as np
k = 10
path_nus = '/home/kiran/Rahul/NUS-WIDE-Lite/'
from sklearn.neighbors import NearestNeighbors
g = open('combined_feature_NUS_Train.pickle','rb')
features = pickle.load(g)
g.close()
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features)
s = len(features)
mu = 4*s
tagvsimage = np.transpose(np.loadtxt(path_nus+'NUS-WIDE-Lite_tags/Lite_Tags1k_Train.txt'))
#print(len(tagvsimage),len(tagvsimage[0]))
#print(tagvsimage[0])
#print(tagvsimage[0])
m = len(tagvsimage)
images_with_tag = []
for j in range(m):
	images_with_tag.append(sum(value > 0 for value in tagvsimage[j])) #Storing and calculating no of images having the tag.
#print(s-len(tagvsimage[0]))
phi_x = []
for i in range(s):
	lrange = nbrs.kneighbors([features[i]],k,return_distance = False)[0]
	phi = np.zeros((k,m))
	#print(phi)
	for l,im in enumerate(lrange):
		#print(im)
		for j in range(m):
			#print(tagvsimage[j][im])
			phi[l][j] = (tagvsimage[j][im]*mu + images_with_tag[j])/(mu + s)
	phi_x.append(np.matrix(phi))
	if i%1000==0:
		print(i)
print('Time to Pickle')
g = open('Phi_x.pickle','wb')
pickle.dump(phi_x,g)
g.close()
		

