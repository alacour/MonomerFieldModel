import numpy as np
from joblib import dump, load
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed


def onedip(p):
    """Compute the dipole vector for a single water molecule.

    No need to rotate the molecule because it assign a charge to each atom. The orientation
    is captured by that atom's orientations.

    Parameters:
        p: (3, 3) array of atomic positions [O, H1, H2].

    Returns:
        (3,) dipole vector in the lab frame.
    """
    vec_aim = np.array([1.0, 0, 0])
    d1 = p[1] - p[0]
    d2 = p[2] - p[0]
    s1 = np.linalg.norm(d1)
    s2 = np.linalg.norm(d2)
    npos = np.array([[0, 0, 0], d1, d2])         
    pr = np.sum(npos[1]*npos[2])/s1/s2
    angle = np.arccos(np.round(pr, 8))*180/np.pi

    qH = 0.333059
    qO = -2*qH
    jb = -0.19008
    ja = 0.0620229
    jbb = -0.0627958
    r0 = 0.959274
    angle0 = 105.0387

    dqH1 = jb * (s1 - r0) + ja * (angle - angle0)*np.pi/180 + jbb * (s2 - r0)
    dqH2 = jb * (s2 - r0) + ja * (angle - angle0)*np.pi/180 + jbb * (s1 - r0)
    dqO = -dqH1 - dqH2

    current_qH1 = qH + dqH1 
    current_qH2 = qH + dqH2
    current_qO = qO + dqO
    dipole = p[0] * current_qO + p[1] * current_qH1 + p[2] * current_qH2
    
    return dipole

def twodip(p, field1, field2, jfield=0.5):
    """Compute the two-body corrected dipole vector for a water molecule.

    Extends the one-body fluctuating charge model with a field-dependent
    charge perturbation that captures the intermolecular (two-body) induced
    dipole from the local electric field environment.

    Parameters:
        p: (3, 3) array of atomic positions [O, H1, H2].
        field1, field2: Electric field projections along each OH bond.
        jfield: Field-charge coupling constant (default 0.1).

    Returns:
        (3,) dipole vector in the lab frame, converted to eA units.
    """
    d1 = p[1] - p[0]
    d2 = p[2] - p[0]
    s1 = np.linalg.norm(d1)
    s2 = np.linalg.norm(d2)
    npos = np.array([[0, 0, 0], d1, d2], dtype='float64')
    pr = np.sum(npos[1]*npos[2])/s1/s2
    angle = np.arccos(np.round(pr, 8))*180/np.pi
    qH = 0.333059
    qO = -2*qH
    jb = -0.19008
    ja = 0.0620229
    jbb = -0.0627958
    r0 = 0.959274
    angle0 = 105.0387
    dqH1 = jb * (s1 - r0) + ja * (angle - angle0)*np.pi/180 + jbb * (s2 - r0) + jfield * field1
    dqH2 = jb * (s2 - r0) + ja * (angle - angle0)*np.pi/180 + jbb * (s1 - r0) + jfield * field2
    dqO = -dqH1 - dqH2

    current_qH1 = qH + dqH1
    current_qH2 = qH + dqH2
    current_qO = qO + dqO

    dipole = p[0] * current_qO + p[1] * current_qH1 + p[2] * current_qH2

    return np.array(dipole, dtype='float64') 


