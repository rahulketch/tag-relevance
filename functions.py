from nltk.corpus import wordnet as wn
from scipy.sparse import *
from scipy.sparse.linalg import norm
import scipy.sparse as sps
import pickle
def frob(A,B):
	return np.trace(((A.getT())*B))

def f(tag,Y,U,v):
	return frob(np.kron(U*U.getT(),v),np.kron(eq(tag),psi(tag,Y)))

def psi(tag,Y,rel_im,non_rel_im):
	sum_ = 0
	for i in rel_im:
		for j in non_rel_im:
			sum = sum + Y[i][j]*(phi(i)-phi(j))
	return sum/(len(rel_im)*(len(non_rel_im)))

def phi(x,k,m):
	phi_ = np.zeroes(k,m)
	for i range(k):
		for j in range(m):
			phi_[i,j] = small_phi(neighbour(x,i),tag[j])
	return phi

def small_phi(z,tag,mu):
	return (mu*delta(z,tag)+images_With_tag(tag))/(mu + total_images)
def func_optimize(v,U,slack,lambda):
	return (lambda/2.0)*((frob(v,v)) + frob(U,U)) + slack

def map_loss(Yground,Y,tag):
	return 1 - AP(Yground,Y,tag)
def AP(Yground,Y):

def getYijTrue(i,j,relevant_set):
	if i in relevant_set:
		if j in relevant_set:
			return 0
		else:
			return 1
	else:
		if j in relevant_set:
			return -1
		else:
			return 0
def getYij(i,j,relevance):
	if relevance[i]>relevance[j]:
		return 1
	if relevance[i]<relevance[j]:
		return -1
	return 0









