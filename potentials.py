import numpy as np
from scipy.optimize import curve_fit, minimize
from joblib import dump, load
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.transform import Rotation as R
from scipy.special import factorial, hermite
from joblib import Parallel, delayed
import os
poly = PolynomialFeatures(12)
pre = lambda x: krr.predict(poly.fit_transform(x))

_DIR = os.path.dirname(os.path.abspath(__file__))
shifts = np.load(os.path.join(_DIR, 'data', 'one_body_shifts.npy'), allow_pickle=True)
krr = load(os.path.join(_DIR, 'data', 'one_body_ridge.joblib'))

def fastonebody(array, field1, field2, kstretch, kb1, kb2):
    """Vectorized one-body energy with electric field perturbations.

    Parameters:
        array: (N, 3) array of [stretch1, stretch2, angle] for each geometry.
        field1, field2: Electric field projections along each OH bond.
        kstretch: Stretch-field coupling constant.
        kb1, kb2: Linear and quadratic bending-field coupling constants.

    Returns:
        (N,) array of perturbed one-body energies.
    """
    inputs = ((array - shifts) / shifts).astype('float64')
    fieldenergy = (-kstretch*(array[:,0]*field1 + array[:,1]*field2) +  -kstretch*((array[:,0] - 0.958929)**2*field1 + (array[:,1] - 0.958929)**2*field2)*0.79/0.37/2
                   + (field1 + field2)*(kb1*(array[:,2] - 104.3636) + kb2*(array[:,2] - 104.3636)**2)
                   + 0.2*(array[:,0] - 0.958929)*(array[:,1] - 0.958929)*(field1 + field2))
    return fieldenergy + krr.predict(poly.fit_transform(inputs))


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

def angle_energy(angle, efield1, efield2,  kb1, kb2):
    """Compute the electric-field-induced perturbation to the HOH bending energy.

    Includes linear and quadratic coupling terms relative to the equilibrium
    angle of 104.3636 degrees.

    Parameters:
        angle: HOH bond angle in degrees.
        efield1, efield2: Electric field projections along each OH bond.
        kb1, kb2: Linear and quadratic bending-field coupling constants.

    Returns:
        Field-induced bending energy perturbation (scalar).
    """
    angle = angle - 104.3636
    return   (efield1 + efield2)*(kb1*angle + kb2*angle**2)



