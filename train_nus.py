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
	return frob(np.kron(np.dot(U,U.transpose()),v),np.kron(eq(tag),psi(tag,Y)))

def eq(tag):
	global query_tag_index1k
	global m
	print(m)
	temp = np.zeros((1,m))
	print(temp)
	temp[0][query_tag_index1k[tag]] = 1.0
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
def func_optimize(v,U,slack,lambda):
	return (lambd/2.0)*((frob(v,v)) + frob(U,U)) + slack
def f_max_constraint(tag,U,v,image):
	global phi
	return frob(np.kron(np.dot(U,U.transpose()),v),np.kron(eq(tag),phi[image]))

# def cutting_plane(U,v,slack,query_tags):
# 	w = []
# 	for tag in query_tags:



#import functions as fn
k = 50 #nearest neighbours
p = 100 #latent space dimension
lamb = 100 # parameter for optimization
m = 1000 #no of tags

f = open('query_tags_NUS.pickle','rb')
query_tags = pickle.load(f) #This would contain the tags we want to train on
f.close()
f = open('Concepts81.txt','r')
all_concepts = f.read().splitlines() #Contains all the concepts, will be used for iterating.
f.close()
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

relevant_images_tag = dict.fromkeys(query_tags) #Ground truth
for query in query_tags:
	temp = np.loadtxt(path_nus+'NUS-WIDE-Lite_groundtruth/Lite_Labels_'+query+'_Train.txt')
	relevant_images_tag[query] = np.where(temp==1)[0]
s = len(temp) #no of images
mu = 4*s
#print(no_of_images)
"""Initialize the quantities"""
U = np.matrix(np.ones((p,m)))
v = np.matrix(np.ones((k,1)))







