import pyemma
import numpy as np
from glob import glob
import mdtraj as md

features = sorted(glob('contacts/*.npy'))
features = pyemma.coordinates.source(features) # this doesn't load all features into memory at the same time
# if not that much data it will be faster to use .load instead of .source

# lag here set to 1 ns for 0.1ns/frame strided data, change accordingly
# using commute_map note in adaptive sampling I used kinetic_map
tica = pyemma.coordinates.tica(features, lag=10, dim=5, kinetic_map=False, commute_map=True)
tica_projection = tica.get_output()

# save tica outputs
np.save('tica_projection.npy', tica_projection)
np.save('feature_tic_correlation.npy', tica.feature_TIC_correlation)

kmeans = pyemma.coordinates.cluster_kmeans(tica_projection, k=100, max_iter=1000, n_jobs=1)
dtrajs = kmeans.dtrajs
np.save('dtrajs.npy', dtrajs)

# lag here set to 50 ns for 0.1ns/frame strided data, change accordingly
msm = pyemma.msm.estimate_markov_model(list(dtrajs), lag=500)

# transition matrix
np.save('transition_matrix.npy', msm.P)
# microstate populations
np.save('micro_populations.npy', msm.pi)

# macrostate identities - use 10 macrostates, change here with informed choices
pcca = msm.pcca(10)

# macrostate populations
np.save('macro_populations.npy', pcca.coarse_grained_stationary_probability)

# convert microstate dtrajs to macrostate labels - note that not all states might be in the msm active set
# those that are not will have nans inserted
micro_to_macro_dict = dict()
for i,state in enumerate(msm.active_set):
    micro_to_macro_dict[state] = pcca.metastable_assignment[i]

for traj in dtrajs:  
    dtraj_macro = []
    for state in traj:
        if state in micro_to_macro_dict:
            dtraj_macro.append(state)
        else:
            dtraj_macro.append(np.nan)
    dtrajs_macro.append(dtraj_macro)

np.save('dtrajs_macro.npy', dtrajs_macro)

