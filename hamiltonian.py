import numpy as np
import scipy.constants as const

def hamiltonian(V,k,dx=None,dy=None):
    if dx==None:
        dx = const.value('Bohr radius')/k
    if dy==None:
        dy = const.value('Bohr radius')/k
        
    H = np.matrix(np.zeros((k**2,k**2)))
    
    for i in range(k**2):
        if i%k!=k-1:
            H[i,i+1]-=1/dx**2
        if i%k!=0:
            H[i,i-1]-=1/dx**2
        if i+k<k**2:
            H[i,i+k]-=1/dy**2
        if i-k>=0:
            H[i,i-k]-=1/dy**2
        H[i,i]+=2/dx**2+2/dy**2+V(i,i,k,dx,dy)
    return H*const.hbar**2/2/const.m_e


def nullPotential(i,j,k):
    return 0

def coloumbPotential(i,j,k,dx,dy):
    m = i%k
    n = i//k
    x0 = (k/2-1/2)*dx
    y0 = (k/2-1/2)*dy
    r = ((m*dx-x0)**2+(n*dy-y0)**2)**0.5
    return 2*const.m_e/const.hbar**2/(4*const.pi*const.epsilon_0)*const.e**2/r



def vectorToMatrix(v):
    k = int(v.size**0.5)
    M = np.matrix(np.zeros((k,k)))
    for i in range(v.size):
        M[i%k,i//k]=v[i]
    return M
