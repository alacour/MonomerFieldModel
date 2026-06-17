import numpy as np
from joblib import dump, load
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
krrpol = load(os.path.join(_DIR, 'data', 'polarizability_krr.joblib')) #The polarizability model


def onepol(p):


    """Compute the lab-frame polarizability tensor for a single water molecule.

    Rotates the molecule into a canonical orientation, predicts the
    polarizability in the molecular frame, then rotates back to the lab frame.

    Parameters:
        p: (3, 3) array of atomic positions [O, H1, H2].

    Returns:
        (3, 3) polarizability tensor in the lab frame.
    """

    polypol = PolynomialFeatures(5)
    vec_aim = np.array([1.0, 0, 0])
    d1 = p[1] - p[0]
    d2 = p[2] - p[0]
    s1 = np.linalg.norm(d1)
    s2 = np.linalg.norm(d2)
 
    npos = np.array([[0, 0, 0], d1, d2])         
    cross = np.cross(d1, d2)
    cross = cross / np.linalg.norm(cross)
    
    add = np.array([0, 0, 1])
    bisect = add + cross   
    
    if (np.linalg.norm(bisect) > 1e-8) * (np.abs(np.linalg.norm(bisect) - 2) > 1e-8): #Guarding against add = -cross
        bisect = bisect / np.linalg.norm(bisect)
        qcross = R.from_quat([bisect[0], bisect[1], bisect[2], 0])
        crossmat = qcross.as_matrix()
        npos = qcross.apply(npos)
    else:
        qcross = R.from_quat([0, 0, 0, 1])
        crossmat = qcross.as_matrix()
        npos = np.copy(npos)
    
    orientation = (npos[1]/s1 + npos[2]/s2)
    orientation = orientation / np.linalg.norm(orientation)
    pr = np.sum(vec_aim * orientation)
    angle = np.arccos(np.round(pr, 8))

    sign = np.sign(np.cross(vec_aim, orientation)[2])
    angle = -angle*sign
    
    if abs(angle) > 1e-8: #Guarding against small angles
        qtwist = R.from_quat([0, 0, np.sin(angle/2), np.cos(angle/2)])
        revtwist = R.from_quat([0, 0, np.sin(-angle/2), np.cos(-angle/2)])
        revtwistmat = revtwist.as_matrix()
        npos = qtwist.apply(npos)

    else:
        qtwist = R.from_quat([0, 0, 0, 1])
        revtwist = qtwist
        twistmat = qtwist.as_matrix()
        revtwistmat = revtwist.as_matrix()
        npos = qtwist.apply(npos)
  
    final_flip = 1
    if npos[2,0] > npos[1,0]:
        final_flip = -1
    
    pr = np.sum(npos[1]*npos[2])/s1/s2
    angle = np.arccos(np.round(pr, 8))*180/np.pi
    inarray = (np.array([[s1, s2, angle]]))
    inarray[:,:2] = np.flip(np.sort(inarray[:,:2], axis=1), axis=1)
    perpol = krrpol.predict(polypol.fit_transform(inarray))[0]

    perpol[[1, 3]] = final_flip*perpol[[1, 3]]
    perpol = np.reshape(perpol, [3, 3])
    inverse_mat = np.matmul(crossmat, revtwistmat)
    pol = np.matmul(np.matmul(inverse_mat, perpol), np.linalg.inv(inverse_mat))
    
    return pol
