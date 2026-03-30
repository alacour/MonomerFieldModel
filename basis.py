import numpy as np
from scipy.special import factorial, hermite

def basis3(n1, n2, n3, x, y, z, mul1, mul2, mul3):
    
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


def numerical_derivative(i, j, k, x, y, z, h12,  mul1, mul2, mul3):
    
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

