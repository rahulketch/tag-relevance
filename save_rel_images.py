
# In[1]:

from nltk.corpus import wordnet as wn
import scipy.sparse as sps
import pickle
path = "../mirflickr/meta/tags/"
no_of_images = 25000


# In[2]:

#This block preprocesses to get list of tags which are in more than 50 images and are in wordnet database.
#no_of_images = 25000
all_tags = dict.fromkeys([])
for i in range(no_of_images):
    with open(path+'tags{}.txt'.format(i+1)) as f:
        file_tags = f.read().splitlines() 
        for tag in file_tags:
            if tag in all_tags:
                all_tags[tag] = all_tags[tag] + 1
            elif wn.synsets(tag.decode('utf-8')):
                all_tags[tag] = 1
rel_tags = []
for tag in all_tags:
    if all_tags[tag]>=50:
        rel_tags.append(tag)

                


# In[5]:

#Making the ground truth for tags. 
image_rel_q = dict.fromkeys([])
#image_nonrel_q = dict.fromkeys(rel_tags)
for i in range(no_of_images):
    with open(path+'tags{}.txt'.format(i+1)) as f:
        file_tags = f.read().splitlines()
        for tag in file_tags:
            if tag in rel_tags:
                if tag in image_rel_q:
                    image_rel_q[tag].append(i)
                else:
                    image_rel_q[tag] = []
                    image_rel_q[tag].append(i)



for tag in rel_tags:
    print(tag)
    print(len(image_rel_q[tag]))
    f = open('../relevant_images/{}'.format(tag),'wb')
    pickle.dump(image_rel_q[tag],f)
    f.close()

            
            