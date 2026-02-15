import numpy as np
from scipy.optimize import curve_fit, minimize
from joblib import dump, load
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import *
from time import time,sleep
from scipy.spatial.transform import Rotation as R
from scipy.special import factorial, hermite
from joblib import Parallel, delayed

shifts = np.load('one_body_shifts.npy', allow_pickle=True)
krr = load('one_body_ridge.joblib')
poly = PolynomialFeatures(12)
pre = lambda x: krr.predict(poly.fit_transform(x))

def onebody(p):
    """Predict the one-body potential energy for a single water molecule.

    Parameters:
        p: (3, 3) array of atomic positions [O, H1, H2].

    Returns:
        Predicted energy from the KRR model (scalar).
    """
    d1 = p[1] - p[0]
    d2 = p[2] - p[0]
    stretch1 = np.linalg.norm(d1)
    stretch2 = np.linalg.norm(d2)
    pr = np.sum(d1*d2)/stretch1/stretch2
    angle = np.arccos(np.round(pr, 8)) *180/np.pi

    ar = ((np.array([[stretch1, stretch2, angle]]) - shifts)/shifts).astype('float64')
    return krr.predict(poly.fit_transform(ar))[0]


def fastonebody(array, field1, field2, bendfield1, bendfield2, kstretch, kb1, kb2):
    """Vectorized one-body energy with electric field perturbations.

    Parameters:
        array: (N, 3) array of [stretch1, stretch2, angle] for each geometry.
        field1, field2: Electric field projections along each OH bond.
        bendfield1, bendfield2: Electric field components for bending.
        kstretch: Stretch-field coupling constant.
        kb1, kb2: Linear and quadratic bending-field coupling constants.

    Returns:
        (N,) array of perturbed one-body energies.
    """
    inputs = ((array - shifts) / shifts).astype('float64')
#    print(field1, bendfield1)
    fieldenergy = (-kstretch*(array[:,0]*field1 + array[:,1]*field2) +  -kstretch*((array[:,0] - 0.958929)**2*field1 + (array[:,1] - 0.958929)**2*field2)*0.79/0.37/2
                   + (bendfield1 + bendfield2)*(kb1*(array[:,2] - 104.3636) + kb2*(array[:,2] - 104.3636)**2))
    return fieldenergy + krr.predict(poly.fit_transform(inputs))

krrpol = load('polarizability_krr.joblib')

def prepol(x):
    """Predict polarizability tensor elements in the molecular frame.

    Sorts stretch distances in descending order before prediction.

    Parameters:
        x: (N, 3) array of [stretch1, stretch2, angle].

    Returns:
        Predicted polarizability tensor elements from the KRR model.
    """
    x[:,:2] = np.flip(np.sort(x[:,:2], axis=1), axis=1)
    return krrpol.predict(polypol.fit_transform(x))

polypol = PolynomialFeatures(5)


def onepol(p):
    """Compute the lab-frame polarizability tensor for a single water molecule.

    Rotates the molecule into a canonical orientation, predicts the
    polarizability in the molecular frame, then rotates back to the lab frame.

    Parameters:
        p: (3, 3) array of atomic positions [O, H1, H2].

    Returns:
        (3, 3) polarizability tensor in the lab frame.
    """
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
    
    if (np.linalg.norm(bisect) > 1e-8) * (np.abs(np.linalg.norm(bisect) - 2) > 1e-8):
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
    
    if abs(angle) > 1e-8:
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
#    print('pold1', s1, 'd2', s2, 'd3', angle)
    inarray = (np.array([[s1, s2, angle]]))# - shifts)/shifts
    
    perpol =  prepol(inarray)[0]
    perpol[[1, 3]] = final_flip*perpol[[1, 3]]
    perpol = np.reshape(perpol, [3, 3])
    inverse_mat = np.matmul(crossmat, revtwistmat)
    pol = np.matmul(np.matmul(inverse_mat, perpol), np.linalg.inv(inverse_mat))

    
    return pol

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
#    print(current_qH1, current_qH2, current_qO)
    dipole = p[0] * current_qO + p[1] * current_qH1 + p[2] * current_qH2
    
    return dipole

def twodip(p, field1, field2, jfield=0.1):
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
 #    print('angled1', s1, 'd2', s2, 'd3', angle)
 #    print(p)
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
     #print(s1, s2, angle, dqH1, dqH2, s1 - r0, s2 - r0, angle-angle0)
     #print(current_qO, current_qH1, current_qH2)
    
     dipole = p[0] * current_qO + p[1] * current_qH1 + p[2] * current_qH2

     return np.array(dipole, dtype='float64') * 4.803

def minima_search(efield1, efield2, bendfield1, bendfield2, min1, min2, min3, kstretch, kb1, kb2):
    """Find the equilibrium geometry of water under an applied electric field.

    Minimizes the field-perturbed potential energy surface with respect to
    the two OH stretches and the HOH angle.

    Parameters:
        efield1, efield2: Electric field projections along each OH bond.
        bendfield1, bendfield2: Electric field components for bending.
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
        perturbed = perturbed + angle_energy(a3, efield1, efield2, bendfield1, bendfield2, kb1, kb2)

        return perturbed
        
    mini = minimize(elambda, 
                    x0 = [min1, min2, min3], 
                    method = 'L-BFGS-B', 
                    options = {"eps":1e-7, "maxiter":int(1e4)}, 
                    tol=1e-20)
    
    
    #print(mini)
    #print(efield1, efield2)
    return mini.x

def stretch_energy(stretch1, stretch2, efield1, efield2, kstretch):
    """Compute the electric-field-induced perturbation to OH stretch energies.

    Includes linear and quadratic coupling terms relative to the equilibrium
    bond length of 0.958929 angstroms.

    Parameters:
        stretch1, stretch2: OH bond lengths in angstroms.
        efield1, efield2: Electric field projections along each OH bond.
        kstretch: Stretch-field coupling constant.

    Returns:
        Field-induced stretch energy perturbation (scalar).
    """
    stretch1 = stretch1 - 0.958929
    stretch2 = stretch2 - 0.958929
    return   -kstretch*(efield1*stretch1 + efield2*stretch2) +  -kstretch*(efield1*stretch1**2 + efield2*stretch2**2)*0.79/0.37/2

def angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2):
    """Compute the electric-field-induced perturbation to the HOH bending energy.

    Includes linear and quadratic coupling terms relative to the equilibrium
    angle of 104.3636 degrees.

    Parameters:
        angle: HOH bond angle in degrees.
        efield1, efield2: Electric field projections along each OH bond.
        bendfield1, bendfield2: Electric field components for bending.
        kb1, kb2: Linear and quadratic bending-field coupling constants.

    Returns:
        Field-induced bending energy perturbation (scalar).
    """
    angle = angle - 104.3636
    return   (bendfield1 + bendfield2)*(kb1*angle + kb2*angle**2)

def construct_mat(Opos, H1pos, H2pos, efield1, efield2, bendfield1, bendfield2, kstretch, kb1, kb2):
    """Build the 9x9 Cartesian Hessian matrix via finite differences.

    Uses a four-point central difference scheme on the field-perturbed
    potential energy surface.

    Parameters:
        Opos, H1pos, H2pos: (3,) arrays for O, H1, H2 Cartesian positions.
        efield1, efield2: Electric field projections along each OH bond.
        bendfield1, bendfield2: Electric field components for bending.
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
            angleterm = angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2)
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
            angleterm = angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2)
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
            angleterm = angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2)
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
            angleterm = angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2)
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

def eigenvalues(field, kstretch, kb1, kb2, bendfield):
    """Compute harmonic vibrational frequencies and normal modes for one water molecule.

    Finds the field-perturbed equilibrium geometry, constructs and mass-weights
    the Hessian, then diagonalizes to obtain normal mode frequencies and vectors.

    Parameters:
        field: (2,) array of electric field projections [OH1, OH2].
        kstretch: Stretch-field coupling constant.
        kb1, kb2: Linear and quadratic bending-field coupling constants.
        bendfield: (2,) array of bending field components [H1, H2].

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
    bendfield1 = bendfield[0]
    bendfield2 = bendfield[1]

    f = minima_search(field1, field2, bendfield1, bendfield2, 1, 1, 100, kstretch, kb1, kb2)
    stretch1, stretch2, thetastart = f
    Opos = np.array([0, 0, 0])
    H1pos = np.array([1, 0, 0])*stretch1
    H2pos = np.array([np.cos(thetastart/180*np.pi), np.sin(thetastart/180*np.pi), 0])*stretch2
    mat = construct_mat(Opos, H1pos, H2pos, field1, field2, bendfield1, bendfield2, kstretch, kb1, kb2)
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
        
    eig_contain =([field1, field2, bendfield1, bendfield2,
                        freq[0], freq[1], freq[2],
                        ])
    minpos = np.array([Opos, H1pos, H2pos])

    return umasses, eig_contain, vectors, minpos



def extract_coordinates(poses):
    """Convert Cartesian water geometries to internal coordinates.

    Parameters:
        poses: (N, 3, 3) array of atomic positions [O, H1, H2].

    Returns:
        (N, 3) array of [stretch1, stretch2, angle_degrees].
    """
    s1 = poses[:,1] - poses[:,0]
    s2 = poses[:,2] - poses[:,0]
    stretch1 = np.linalg.norm(s1, axis=1)
    stretch2 = np.linalg.norm(s2, axis=1)
    pr = np.sum(s1*s2, axis=1)/stretch1/stretch2
    angle = np.arccos(np.round(pr, 8))*180/np.pi
    return np.array([stretch1, stretch2, angle]).T


def basis3(n1, n2, n3, x, y, z, mul1, mul2, mul3):
    """Evaluate a 3D harmonic oscillator basis function (product of Hermite-Gaussians).

    Parameters:
        n1, n2, n3: Quantum numbers for each dimension.
        x, y, z: Coordinate arrays at which to evaluate.
        mul1, mul2, mul3: Width parameters for each dimension.

    Returns:
        Array of basis function values at the given grid points.
    """
    f = ((mul1*mul2*mul3/np.pi**3)**(1/4))
    
    f = f/np.sqrt(2**n1*factorial(n1)*
                  2**n2*factorial(n2)*
                  2**n3*factorial(n3))
    
    f = f*((np.exp(-mul1/2*x**2))*
           (np.exp(-mul2/2*y**2))*
           (np.exp(-mul3/2*z**2)))
    
    f = f*(hermite(n1)(np.sqrt(mul1)*(x))*
           hermite(n2)(np.sqrt(mul2)*(y))*
           hermite(n3)(np.sqrt(mul3)*(z)))
    
    
    return f


def numerical_derivative(i, j, k, x, y, z, h12, mul1, mul2, mul3):
    """Compute the kinetic energy operator acting on a 3D basis function via finite differences.

    Uses central differences to approximate the second derivative along each
    coordinate, scaled by hbar^2/2.

    Parameters:
        i, j, k: Quantum numbers for each dimension.
        x, y, z: Coordinate arrays at which to evaluate.
        h12: Product of hbar values (hbar1 * hbar2) for unit conversion.
        mul1, mul2, mul3: Width parameters for each dimension.

    Returns:
        Tuple of (psi, d2dx, d2dy, d2dz) where psi is the basis function
        and d2dx/y/z are the kinetic energy contributions along each axis.
    """
    ddd = 1e-5
    psi = basis3(i, j, k, x, y, z, mul1, mul2, mul3)



    fpsi = basis3(i, j, k,  x + ddd, y, z,  mul1, mul2, mul3)
    rpsi = basis3(i, j, k,  x - ddd, y, z,  mul1, mul2, mul3)
    dxpsi = (fpsi + rpsi - 2*psi)/(ddd)**2*h12/2
    
    fpsi = basis3(i, j, k, x, y+ddd, z,  mul1, mul2, mul3)
    rpsi = basis3(i, j, k, x, y-ddd, z,  mul1, mul2, mul3)
    dypsi = (fpsi + rpsi - 2*psi)/(ddd)**2*h12/2
    
    fpsi = basis3(i, j, k, x, y, z+ddd,  mul1, mul2, mul3)
    rpsi = basis3(i, j, k, x, y, z-ddd,  mul1, mul2, mul3)
    dzpsi = (fpsi + rpsi - 2*psi)/(ddd)**2*h12/2
    
    
    #print(psi, fpsi, rpsi)
    return psi, dxpsi, dypsi, dzpsi

