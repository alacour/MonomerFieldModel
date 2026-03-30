#from necessary_functions import *
from potentials import fastonebody
from polarizability import onepol
from dipole import onedip, twodip
from normal_modes import *
from geometry import *
from basis import numerical_derivative
from constants import R0, A0, MO, MH
import numpy as np
import os
import json

def compute_distribution(projs, param_file):#, k1, kb1, kb2, kbendstd):

    """Compute Raman and IR spectral activities for a set of water molecules.

    Solves the anharmonic vibrational Schrodinger equation on a 3D normal-mode
    grid and computes Raman transition polarizabilities and IR transition dipoles.

    Parameters:
        projs: (N, 2) array of electric field projections along each OH bond (V/A).
        k1: Stretch-field coupling constant.
        kb1, kb2: Linear and quadratic bending-field coupling constants.
        kbendstd: Std dev of Gaussian multipliers applied to bending constants.

    Returns:
        0 on completion. Results saved to .npy files (qfreqs, activities, etc.).
    """
    with open(param_file) as f:
        params = json.load(f)
    k1, k2, kb1, kb2, kbstd  = params['k1'], params['k2'], params['kb1'], params['kb2'], params['kbstd']
    processors, nbasis = params['processors'], params['nbasis']
    mul1, mul2, mul3, gridpoints = params['mul1'], params['mul2'], params['mul3'], params['gridpoints']
    m1 = MO
    m2 = MH
    m3 = MH
    hbar1 = 6.5821 * 10**-16
    hr1 = hbar1*2*np.pi
    hbar2 = hbar1 * 1.602*10**-19 * 6.022*10**26 * 10**20
    h2 = hbar2*2*np.pi
    h12 = hbar1 * hbar2
    fields = np.copy(projs)
    muls = np.random.normal(1.0, kbstd, len(fields))  # multipliers for bending energy
    eig_contain = []
    allumasses = []
    harvecs = []
    minposes = []
    mfields = []


    ###### Getting the Harmonic Frequencies
    for i,field in enumerate(fields):
        g = eigenvalues(field, k1, kb1*muls[i], kb2*muls[i])
        if np.sum(np.isnan(g[2])) < 1:
            allumasses.append(g[0])
            eig_contain.append(g[1])
            harvecs.append(g[2])
            minposes.append(g[3])
            mfields.append(field)
    eig_contain = np.asarray(eig_contain)   

    ###### Getting the Anharmonic Frequencies
    print("Computing Frequencies")
    # First we need to compute the grid
    basis0 = 0.0
    basisL = 1.1
    xmin = basis0 - basisL/2
    xmax = basis0 + basisL/2
    xx = np.linspace(xmin, xmax, gridpoints)
    stepsize = xx[1] - xx[0]
    vol = (basisL + stepsize)**3
    xxgrid = np.array([np.repeat(xx, len(xx)**2),
                       np.repeat(xx.tolist()*len(xx), len(xx)),
                       xx.tolist()*len(xx)**2]).T

    energies = []
    vecadds = []
    egrids = []
    polgrids = []
    onedipgrids = []
    twodipgrids = []


    t =time()
    sleep(10)

    def fill_grids(j):
        vectors = harvecs[j]
        poses = minposes[j] 
        poses = np.array(len(xx)**3*[poses])
        field = mfields[j]
        field1 = field[0]
        field2 = field[1]
        
        mul = muls[j]
        umasses = []
        for i in range(3):
            vecadd = np.reshape(np.copy(vectors[:,-3+i]), [3, 3])
            nkeep = np.copy(vecadd)
            vecadd[0] = vecadd[0]/m1**(1/2)
            vecadd[1] = vecadd[1]/m2**(1/2)
            vecadd[2] = vecadd[2]/m3**(1/2)


            vecadd = vecadd / np.linalg.norm(vecadd)
            vecadds.append(vecadd)
            dkeep = np.copy(nkeep)
            dkeep[0] = dkeep[0]**2/m1
            dkeep[1] = dkeep[1]**2/m2
            dkeep[2] = dkeep[2]**2/m3



            umasses.append(1/np.sum(dkeep))
            
        vecadd1 = np.array(len(xx)**3*[vecadds[0]])
        vecadd2 = np.array(len(xx)**3*[vecadds[1]])
        vecadd3 = np.array(len(xx)**3*[vecadds[2]])


        posgrid = poses + (((xxgrid[:,0]*vecadd1.T).T) + 
                           (xxgrid[:,1]*vecadd2.T).T + 
                           (xxgrid[:,2]*vecadd3.T).T)
        cgrid = extract_coordinates(posgrid)
        egrid = fastonebody(cgrid, field1, field2, k1, kb1*mul, kb2*mul)
        polgrid = []
        onedipgrid = []
        twodipgrid = []

        for pos in posgrid:
            pol = onepol(pos)
            polgrid.append(pol)
            dip = onedip(pos)
            onedipgrid.append(dip)
            dip = twodip(pos, field1, field2)
            twodipgrid.append(dip)

        return egrid, polgrid, onedipgrid, twodipgrid, umasses

    grids = Parallel(n_jobs=processors, verbose=10)(delayed(fill_grids)(i) for i in range(len(harvecs[:])))
    sleep(20)
    
    for i in range(len(grids)):
        egrids.append(grids[i][0])
        polgrids.append(grids[i][1])
        onedipgrids.append(grids[i][2])
        twodipgrids.append(grids[i][3])
        allumasses.append(grids[i][4])


    #Now we need to get the wavefunction and its derivative at every gridpoint
    psis = []
    d2d2xs = []
    d2d2ys = []
    d2d2zs = []

    for i in range(0, nbasis):
        for k in range(0, nbasis):
            for m in range(0, nbasis):
                psi1, d2d2x, d2d2y, d2d2z = numerical_derivative(i, k, m, 
                                                        xxgrid[:,0], 
                                                        xxgrid[:,1], 
                                                        xxgrid[:,2], 
                                                        h12, 
                                                        mul1,
                                                        mul2,
                                                        mul3) 
                psis.append(psi1)
                d2d2xs.append(d2d2x)
                d2d2ys.append(d2d2y)
                d2d2zs.append(d2d2z)

    psis = np.array(psis)
    d2d2xs = np.asarray(d2d2xs)
    d2d2ys = np.asarray(d2d2ys)
    d2d2zs = np.asarray(d2d2zs)
    vol = (basisL + stepsize)**3

    def compute_hamiltonians(kt):
        umasses = allumasses[kt]
        egrid = egrids[kt]
        es = []
        for it, psi1 in enumerate(psis[:]):
            d2d2x = d2d2xs[it]
            d2d2y = d2d2ys[it]
            d2d2z = d2d2zs[it]
            jt = 0
            energies = []
            for jt, psi2 in enumerate(psis[:]):
                dx = d2d2x/umasses[0]
                dy = d2d2y/umasses[1]
                dz = d2d2z/umasses[2]
                inte1 = np.average(psi2*(dx + dy + dz))*vol
                inte2 = np.average(psi2*psi1*egrid)*vol
                energies.append(-inte1+inte2)
            es.append(energies)
        return es

    ematrices = Parallel(n_jobs=processors, verbose=10)(delayed(compute_hamiltonians)(i) for i in range(len(harvecs[:])))
    sleep(20)

    # And getting the eigenvalues, eigenvectors
    reseigs = []
    resvecs = []
    for i in range(len(egrids)):
        ees = ematrices[i]
        eigs, vecs = np.linalg.eigh(ees)
        reseigs.append(eigs)
        resvecs.append(vecs)

    reseigs = np.asarray(reseigs)
    resvecs = np.asarray(resvecs)
    reseigs = np.asarray(reseigs)
    resvecs = np.asarray(resvecs)
    freqs = (reseigs[:,1:].T - reseigs[:,0]).T
    freqs = freqs[:,:4] * 8065.544 # Going to cm^-1

    relvecs = resvecs[:,:,0:5]# We only care about the 5 lowest -  the ground state the first four excited states

    t = time()
    activities = []

    for it,vecs in enumerate(resvecs):
        polgrid = np.asarray(polgrids[it])
        onedipgrid = np.asarray(onedipgrids[it])
        twodipgrid = np.asarray(twodipgrids[it])
        psi0 = np.sum(psis.T * vecs[:,0], axis=1).T

        activity = [[],[],[],[]]
        for i in range(4):
            psij = np.sum(psis.T * vecs[:,i+1], axis=1).T
            pol1 = np.average(polgrid.T * psi0 * psij, axis = 2).T*vol
            sym = np.sum(np.diag(pol1))**2# / 9
            beta = np.copy(pol1)
            beta_trace = np.sum(np.diag(beta))
            beta -= beta_trace / 3 * np.diag([1, 1, 1])
            anti = np.sum(np.diag(np.matmul(beta, beta))) * 21 / 90 
            activity[0].append(sym)
            activity[1].append(anti)
            dip1 = np.average(onedipgrid.T * psi0 * psij, axis = 1).T*vol
            dip1 = np.sum(dip1**2)
            activity[2].append(dip1)
            dip2 = np.average(twodipgrid.T * psi0 * psij, axis = 1).T*vol
            dip2 = np.sum(dip2**2)
            activity[3].append(dip2)

        activities.append(np.concatenate(activity))

    activities = np.array(activities)
    print(activities[0])
    np.savez('results.npz', freqs=freqs, 
                            activities=activities, 
                            fields=fields,
                            eigenvalues=eigs,
                             )
    return 0 


