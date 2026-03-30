import numpy as np


def extract_coordinates(poses):
    s1 = poses[:,1] - poses[:,0]
    s2 = poses[:,2] - poses[:,0]
    stretch1 = np.linalg.norm(s1, axis=1)
    stretch2 = np.linalg.norm(s2, axis=1)
    pr = np.sum(s1*s2, axis=1)/stretch1/stretch2
    angle = np.arccos(np.round(pr, 8))*180/np.pi
    return np.array([stretch1, stretch2, angle]).T


