from nltk.corpus import wordnet as wn
import scipy.sparse as sps
import pickle
path = "../mirflickr/meta/tags/"
no_of_images = 25000


# In[2]:

#This block preprocesses to get list of tags which are in more than 50 images and are in wordnet database.
#no_of_images = 25000
# all_tags = dict.fromkeys([])
# for i in range(no_of_images):
#     with open(path+'tags{}.txt'.format(i+1)) as f:
#         file_tags = f.read().splitlines() 
#         for tag in file_tags:
#             if tag in all_tags:
#                 all_tags[tag] = all_tags[tag] + 1
#             elif wn.synsets(tag.decode('utf-8')):
#                 all_tags[tag] = 1
# rel_tags = []
# for tag in all_tags:
#     if all_tags[tag]>=50:
#         rel_tags.append(tag)
"""Checking if the concept is present in the 1k tags and writing them to the file for future reference"""
f = open('Concepts81.txt','r')
concepts = f.read().splitlines()
f.close()
f = open('TagList1k.txt','r')
tags = f.read().splitlines()
f.close()
count = 0
query_tags = []
for concept in concepts:
    if concept in tags:
        query_tags.append(concept)
		#print(concept)
        count+=1
print(query_tags)
print(count,len(query_tags))
g = open('query_tags_NUS.pickle','wb')
pickle.dump(query_tags,g)
g.close()
