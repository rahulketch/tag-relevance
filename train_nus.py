#from nltk.corpus import wordnet as wn
#from scipy.sparse import *
#from scipy.sparse.linalg import norm
path_nus = '/home/kiran/Rahul/NUS-WIDE-Lite/'
import scipy.sparse as sps
import pickle
import numpy as np
def frob(A,B):
	return np.trace(np.dot(A.transpose(),B))
def f(tag,Y,U,v):
	return frob(np.kron(np.dot(U.transpose(),U),v),np.kron(eq(tag),psi(tag,Y)))

def eq(tag):
	global query_tag_index1k
	global m
	#print(m)
	temp = np.zeros((m,1))
	#print(temp)
	temp[query_tag_index1k[tag]][0] = 1.0
	return np.matrix(temp)
def psi(tag,Y):
	global phi
	global relevant_images_tag
	global s
	sum = 0.0*phi[0]
	for i in relevant_images_tag[tag]:
		for j in range(s):
			if j not in relevant_images_tag[tag]:
				sum = sum + Y[i][j]*(phi[i]-phi[j])/(len(relevant_images_tag[tag])*(s-len(relevant_images_tag[tag])))
	return sum
def func_optimize(v,U,slack,lambd):
	return (lambd/2.0)*((frob(v,v)) + frob(U,U)) + slack
def f_max_constraint(tag,U,v,image):
	global phi
	return frob(np.kron(np.dot(U.transpose(),U),v),np.kron(eq(tag),phi[image]))
def f_max_constraint2(A,B,image):
	global phi
	return frob(A,np.kron(B,phi[image]))

def find_max_constraint(tag,U,v):
	global relevant_images_tag
	global not_relevant_image_tag
	global m
	global s
	global phi
	print('lenght',s-len(relevant_images_tag[tag]) - len(not_relevant_image_tag[tag]),len(relevant_images_tag[tag]),len(not_relevant_image_tag[tag]))
	A = np.kron(np.dot(U.transpose(),U),v)
	B = eq(tag)
	# scores_relevant = []
	# scores_not_relevant = []
	# print('before vectorize')
	# f_max_vectorize = np.vectorize(f_max_constraint)
	# print('after vectorize')
	scores_relevant = [f_max_constraint2(A,B,i) for i in relevant_images_tag[tag]]
	# for i in relevant_images_tag[tag]:
	# 	scores_relevant.append(f_max_constraint(tag,U,v,i))
	#scores_relevant = f_max_vectorize(tag,U,v,relevant_images_tag[tag])
	print("After Relevant")
	# for i in not_relevant_image_tag[tag]:
	# 	scores_not_relevant.append(f_max_constraint(tag,U,v,i))
	scores_not_relevant = [f_max_constraint2(A,B,i) for i in not_relevant_image_tag[tag]]
	#scores_not_relevant = f_max_vectorize(tag,U,v,not_relevant_image_tag[tag])
	print("After not relevant")
	Cx = zip(relevant_images_tag[tag],scores_relevant).sort(key = lambda x: x[1])
	print("Done CX")
	Cx_bar = zip(not_relevant_image_tag[tag],scores_not_relevant).sort(key = lambda x: x[1])
	print(Cx)
	print(Cx_bar)
	return 0

		





# def cutting_plane(U,v,slack,query_tags):
# 	w = []
# 	for tag in query_tags:



#import functions as fn
k = 10 #nearest neighbours
p = 100 #latent space dimension
lamb = 100 # parameter for optimization
m = 1000 #no of tags

f = open('query_tags_NUS.pickle','rb')
query_tags = pickle.load(f) #This would contain the tags we want to train on
f.close()
f = open('Concepts81.txt','r')
all_concepts = f.read().splitlines() #Contains all the concepts, will be used for iterating.
f.close()
g = open('Phi_x.pickle','rb')
phi = pickle.load(g)
g.close()
print('Done with loading')
# print(all_concepts)
# print(len(all_concepts))
query_tag_index = dict.fromkeys(query_tags)
for i,tag in enumerate(all_concepts):
	if tag in query_tags:
		query_tag_index[tag] = i
		#print(tag,i)

f = open('TagList1k.txt','r')
all_tags = f.read().splitlines() #Contains all the tags, will be used for iterating.
f.close()
# print(all_concepts)
# print(len(all_concepts))
query_tag_index1k = dict.fromkeys(query_tags)
for i,tag in enumerate(all_tags):
	if tag in query_tags:
		query_tag_index1k[tag] = i
		#print(tag,i)
print('Before sorting images')
relevant_images_tag = dict.fromkeys(query_tags) #Ground truth
not_relevant_image_tag = dict.fromkeys(query_tags)
for query in query_tags:
	temp = np.loadtxt(path_nus+'NUS-WIDE-Lite_groundtruth/Lite_Labels_'+query+'_Train.txt')
	relevant_images_tag[query] = np.where(temp==1)[0]
	not_relevant_image_tag[query] = np.where(temp==0)[0]
s = len(temp) #no of images
print('images are done')
mu = 4*s
#print(no_of_images)
"""Initialize the quantities"""
U = np.matrix(np.ones((p,m)))
v = np.matrix(np.ones((k,1)))
W = []
while True:
	Y_q = []
	for tag in query_tags:
		Y_q.append(find_max_constraint(tag,U,v))
		break
	break







