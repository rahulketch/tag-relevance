import pickle
import numpy as np
print('Before Load')
g = open('Phi_x.pickle','rb')
phi = pickle.load(g)
g.close()
print('Done with loading')