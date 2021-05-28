# run this in a directory called contacts/ which sits one level below the directory containing h5s of data

import numpy as np
from glob import glob
import mdtraj as md
import multiprocessing

n_threads = 16
stride = 10

# uncomment this line and another in get_contacts and provide array of residue indexes, 
# e.g. contact_indexes = [[0,1], [1,2], ...] for subset of contacts, otherwise calc. all
# contact_indexes = np.load('contact_indexes.npy')

def get_contacts(traj):
    traj_ = md.load(traj, stride=stride)
    contacts = md.compute_contacts(traj_)[0]
    #contacts = md.compute_contacts(traj_, contacts=contact_indexes)[0]
    split = traj.split('/')
    np.save(f'{split[1][:-3]}.npy', contacts)

pool = multiprocessing.Pool(n_threads)
trajs = sorted(glob(f'../run{run}-clone*.h5'))
pool.map(get_contacts, trajs)