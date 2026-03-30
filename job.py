import numpy as np

from calculate import compute_distribution

ii, nrandos, leng, end =  0, 3000, 2.0, 2.0


hprojs = np.load('amoeba_pair_fields.npy', allow_pickle=True)  # Fields experienced by Hs
interface_dis = np.load('Odisses.npy', allow_pickle=True) # Distance to WCI
oilangles  = np.load('amoeba_angles.npy', allow_pickle=True)  # Hydrogen bonding angles


Oposes = oilangles[ii][:,0]  # Zposition of oxygens
projs = hprojs[ii] 
angles = oilangles[ii][:,1:]

alli = interface_dis[ii]
projs = projs[Oposes > 0]  #  Only computed WCI distances for waters with Zpositions > 0
angles = angles[Oposes > 0]

lenf = len(projs)
randos = np.argsort(np.random.uniform(0, 1, lenf))[:nrandos] # Random subset
alli = alli[randos]
projs = projs[randos]
angles = angles[randos]

bound = 90

cond =  (alli > end) *  (alli < end + leng) # only getting waters within a certain distance from the WCI
projs = projs[cond]*14.4 #Converting to V/A
angles = angles[cond]


cond = (angles[:,0] > bound) + (angles[:,1] > bound) # Making Sure I only get waters with both  OH's bonded - I edit here 2026
projs = projs[cond]
angles = angles[cond]


projs = np.linspace(0.0, 4.0, 2)
projs = np.array([projs]*2).T

compute_distribution(projs, 'amoeba_params.json')#,  0.41730000000000006, -0.9*0.00107, 0.0*2.996e-05, 0.0*0.37)

np.save('used', randos)
