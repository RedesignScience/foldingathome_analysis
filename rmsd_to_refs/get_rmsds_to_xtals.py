from glob import glob
import mdtraj as md
import numpy as np
import multiprocessing

n_jobs = 32
superpose_string = 'protein and (resid < 31 or resid > 48) and name CA'
rmsd_string = 'resid >= 31 and resid <= 48 and not element H'

pdbs = sorted(glob('/home/rafal.wiewiora/repos/BCL_SETUP/output/*/solvated.pdb'))
refs = [md.load(pdb) for pdb in pdbs]
superpose_selection = refs[0].top.select(superpose_string)
rmsd_selection = refs[0].top.select(rmsd_string)

def calc_rmsd(traj):
    try: # some XTCs are failing
        traj = md.load(traj, top='top.pdb')
    except:
        return None
    traj = traj.image_molecules()
    rmsds = []
    for ref in refs:
        traj = traj.superpose(ref, atom_indices=superpose_selection)
        rmsd = np.sqrt(3*np.mean((traj.xyz[:, rmsd_selection, :] - ref.xyz[:, rmsd_selection, :])**2, axis=(1,2)))
        rmsds.append(rmsd)
    return rmsds

pool = multiprocessing.Pool(n_jobs)

for run in range(35):
    trajs = sorted(glob(f'data/PROJ17800/RUN{run}/CLONE*/results*/positions.xtc'))
    rmsds = pool.map(calc_rmsd, trajs)
    np.save(f'run{run}_xtal_rmsds.npy', rmsds)
