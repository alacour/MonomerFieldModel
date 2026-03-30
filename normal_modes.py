import numpy as np
from scipy.optimize import curve_fit, minimize
from joblib import dump, load
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import *
from time import time,sleep
from scipy.spatial.transform import Rotation as R
from scipy.special import factorial, hermite
from joblib import Parallel, delayed
from potentials import *
import os

poly = PolynomialFeatures(12)
pre = lambda x: krr.predict(poly.fit_transform(x))
_DIR = os.path.dirname(os.path.abspath(__file__))
shifts = np.load(os.path.join(_DIR, 'data', 'one_body_shifts.npy'), allow_pickle=True)
krr = load(os.path.join(_DIR, 'data', 'one_body_ridge.joblib'))




def minima_search(efield1, efield2, min1, min2, min3, kstretch, kb1, kb2):
    """Find the equilibrium geometry of water under an applied electric field.

    Minimizes the field-perturbed potential energy surface with respect to
    the two OH stretches and the HOH angle.

    Parameters:
        efield1, efield2: Electric field projections along each OH bond.
        min1, min2, min3: Initial guesses for stretch1, stretch2, and angle.
        kstretch: Stretch-field coupling constant.
        kb1, kb2: Linear and quadratic bending-field coupling constants.

    Returns:
        (3,) array of optimized [stretch1, stretch2, angle].
    """
    def elambda(ass):
        a1, a2, a3 = ass
        evaluates = (ass - shifts)/shifts
        unperturbed = pre(np.array([evaluates]))
        perturbed = unperturbed  + stretch_energy(a1, a2, efield1, efield2, kstretch)
        perturbed = perturbed + angle_energy(a3, efield1, efield2, kb1, kb2)

        return perturbed
        
    mini = minimize(elambda, 
                    x0 = [min1, min2, min3], 
                    method = 'L-BFGS-B', 
                    options = {"eps":1e-7, "maxiter":int(1e4)}, 
                    tol=1e-20)
    
    
    return mini.x

def construct_mat(Opos, H1pos, H2pos, efield1, efield2, kstretch, kb1, kb2):
    """Build the 9x9 Cartesian Hessian matrix via finite differences.

    Uses a four-point central difference scheme on the field-perturbed
    potential energy surface.

    Parameters:
        Opos, H1pos, H2pos: (3,) arrays for O, H1, H2 Cartesian positions.
        efield1, efield2: Electric field projections along each OH bond.
        kstretch: Stretch-field coupling constant.
        kb1, kb2: Linear and quadratic bending-field coupling constants.

    Returns:
        (9, 9) Hessian matrix of second derivatives.
    """
    mat = []
    allvar = np.concatenate([Opos, H1pos, H2pos])
    de = 1e-4

    for i in range(9):
        tmat = []
        for j in range(9):
            tvar = np.copy(allvar)
            tvar[i] += de
            tvar[j] += de
            #print(tvar)
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8))*180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2, kb1, kb2)
            eff = pre(ar)[0] + bendterm + angleterm
            
            tvar[j] -= 2*de
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8))*180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2,  kb1, kb2)
            eff = pre(ar)[0] + bendterm + angleterm
            
            tvar[j] -= 2*de
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8))*180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2, kb1, kb2)
            efb = pre(ar)[0] + bendterm + angleterm
            
            
            tvar[i] -= 2*de
            tvar[j] += 2*de
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8)) *180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2, kb1, kb2)
            ebf = pre(ar)[0] + bendterm + angleterm
            
            tvar[j] -= 2*de
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8)) *180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2, kb1, kb2)
            ebb = pre(ar)[0] + bendterm + angleterm


            dde = (eff + ebb - efb - ebf)/4/de**2

            tmat.append(dde)

        mat.append(tmat)


    mat = np.asarray(mat)
    return mat

def reduced_mass(mat, m1, m2, m3):
    """Mass-weight the Cartesian Hessian matrix.

    Divides each element by sqrt(m_i * m_j) where m_i and m_j are the
    atomic masses corresponding to the Cartesian coordinates.

    Parameters:
        mat: (9, 9) Cartesian Hessian matrix.
        m1, m2, m3: Atomic masses for O, H1, H2 in amu.

    Returns:
        (9, 9) mass-weighted Hessian matrix.
    """
    tmat = np.copy(mat)
    for i in range(9):
        for j in range(9):
            if i < 3:
                mass1 = m1

            elif i - 3 < 3:
                mass1 = m2
                
            else:
                mass1 = m3

            if j < 3:
                mass2 = m1

            elif j - 3 < 3:
                mass2 = m2
            else:
                mass2 = m3
            tmat[i,j] = tmat[i,j]  / mass1**(1/2) / mass2**(1/2)
            
            
    return tmat


def eigenvalues(field, kstretch, kb1, kb2):
    """Compute harmonic vibrational frequencies and normal modes for one water molecule.

    Finds the field-perturbed equilibrium geometry, constructs and mass-weights
    the Hessian, then diagonalizes to obtain normal mode frequencies and vectors.

    Parameters:
        field: (2,) array of electric field projections [OH1, OH2].
        kstretch: Stretch-field coupling constant.
        kb1, kb2: Linear and quadratic bending-field coupling constants.

    Returns:
        Tuple of (umasses, eig_contain, vectors, minpos) where umasses are
        effective masses, eig_contain holds field values and frequencies,
        vectors are the eigenvectors, and minpos is the equilibrium geometry.
    """
    m1 = 15.999
    m2 = 1.007
    m3 = 1.007
    conv = (1*2*np.pi*3*10**10)**(2)*1.66054e-27/10**20*6.242*10**18
    field1 = field[0]
    field2 = field[1]

    f = minima_search(field1, field2, 1, 1, 100, kstretch, kb1, kb2)
    stretch1, stretch2, thetastart = f
    Opos = np.array([0, 0, 0])
    H1pos = np.array([1, 0, 0])*stretch1
    H2pos = np.array([np.cos(thetastart/180*np.pi), np.sin(thetastart/180*np.pi), 0])*stretch2
    mat = construct_mat(Opos, H1pos, H2pos, field1, field2, kstretch, kb1, kb2)
    tmat = reduced_mass(mat, m1, m2, m3)
    eigs, vectors = np.linalg.eigh(tmat)
    freq = (eigs/conv)[6:]**(1/2)
    
    moves = np.reshape(vectors[:,6:], [3, 3, 3])

    tmoves = np.copy(moves)
    umasses = []
    for i in range(3):
        trans = moves[:,:,i] / (np.array([[m1, m2, m3]]).T**(1/2))
        trans = trans / np.linalg.norm(trans)
        tmoves[:,:,i] = trans
        
        dkeep = np.copy(moves[:,:,i])
        dkeep[0] = dkeep[0]**2/m1
        dkeep[1] = dkeep[1]**2/m2
        dkeep[2] = dkeep[2]**2/m3


        umasses.append(1/np.sum(dkeep))
        
    eig_contain =([field1, field2,
                   freq[0], freq[1], freq[2],
                        ])
    minpos = np.array([Opos, H1pos, H2pos])
    return umasses, eig_contain, vectors, minpos



