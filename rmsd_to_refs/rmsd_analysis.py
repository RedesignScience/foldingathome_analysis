import numpy as np
import mdtraj as md
from glob import glob
import os

# assemble trajectory of the references
pdbs = sorted(glob('/home/rafal.wiewiora/repos/BCL17800_SETUP/output/*/solvated.pdb'))
refs = md.load(pdbs[0])
for pdb in pdbs[1:]:
    refs = md.join([refs,md.load(pdb)])
refs_protein = refs.atom_slice(refs.top.select('protein'))
refs_protein.save('rmsd_SELF_trajs/refs.pdb')

# closest to self, extract the trajectories
runs_self_mins = []
runs_self_mins_indexes = []
for i in range(35):
    run = np.load(f'run{i}_xtal_rmsds.npy', allow_pickle=True)
    self_mins = []
    for traj in run:
        if traj is not None:
            self_mins.append(np.min(traj[i]))
        else:
            self_mins.append(np.nan)
    runs_self_mins.append(np.nanmin(self_mins))
    index = np.nanargmin(self_mins)
    trajs = sorted(glob(f'data/PROJ17800/RUN{i}/CLONE*/results*/positions.xtc'))
    os.system(f'cp {trajs[index]} rmsd_SELF_trajs/{i}.xtc')

superpose_string = 'protein and (resid < 31 or resid > 48) and name CA'
rmsd_string = 'resid >= 31 and resid <= 48 and not element H'
superpose_selection = refs[0].top.select(superpose_string)
rmsd_selection = refs[0].top.select(rmsd_string)
# calc rmsds of the closest to self trajectories to each reference
self_rmsds = []
for i in range(35):
    traj = md.load(f'rmsd_SELF_trajs/{i}.xtc', top='top.pdb')
    traj = traj.image_molecules()
    ref = refs[i]
    traj = traj.superpose(ref, atom_indices=superpose_selection)
    rmsd = np.sqrt(3*np.mean((traj.xyz[:, rmsd_selection, :] - ref.xyz[:, rmsd_selection, :])**2, axis=(1,2)))
    self_rmsds.append(rmsd)
    traj[np.argmin(rmsd)].save(f'rmsd_SELF_trajs/{i}_minframe.pdb')

np.save('self_rmsd_mins.npy', runs_self_mins)
np.save('self_rmsd_drawntrajs_mins.npy', self_rmsds)

# merge minframes into one traj
traj = md.load('rmsd_SELF_trajs/0_minframe.pdb')
for i in range(1,35):
    traj = md.join([traj,md.load(f'rmsd_SELF_trajs/{i}_minframe.pdb')])
traj = traj.atom_slice(traj.top.select('protein'))
traj.save('rmsd_SELF_trajs/minframes_traj.pdb')