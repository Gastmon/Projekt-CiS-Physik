import numpy as np
import scipy.constants as const
import scipy.sparse as sp
import math

def hamiltonian(V,k,dx=None,dy=None,sparse=False):
    if dx==None:
        dx = 16*const.value('Bohr radius')/(k-1)
    if dy==None:
        dy = 16*const.value('Bohr radius')/(k-1)
       
    if sparse:
        H = sp.lil_matrix((k**2,k**2))
    else:
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
        m = i%k
        n = i//k
        x = (m-k/2+1/2)*dx
        y = (n-k/2+1/2)*dy
        H[i,i]+=2/dx**2+2/dy**2+V(x,y)
        
    H=H*const.hbar**2/2/const.m_e/const.eV
    
    if sparse:
        return H.tocsc()
    else:
        return H


def nullPotential(x,y,z=0,radial=False):
    return 0

def coloumbPotential(x,y,z=0,radial=False):
    if radial:
        r=x
    else:
        r = np.linalg.norm((x,y,z))
    return -2*const.m_e/const.hbar**2/(4*const.pi*const.epsilon_0)*const.e**2/r


def cumulativeDistributionGenerator(f,boundary,offset=0,maxInput=None):
    if maxInput==None:
        gmax=-f(0-offset)
    else:
        gmax=f(maxInput-offset)
    def fun(i):
        return f(i-offset)/gmax*boundary
    return fun
    
#integral of f(x)=1/(1-tanh^2(x))-1
#f'(0)=0 is problematic (maybe)
def cubicOrderSinusHyperbolicus(x):
    return 1/4*np.sinh(2*x)-0.5*x
    
#integral of f(x)=1    
def linear(x):
    return x
  
#integral of f(x)=x**2+1000
def cubic(x):
    return 1/3*x**3+1000*x

def hamiltonianWithDistribution(V,k,i,dxDist=linear,dyDist=linear,sparse=False):
    dxDist = cumulativeDistributionGenerator(dxDist, i/2*const.value('Bohr radius'), k/2-1/2)
    dyDist = cumulativeDistributionGenerator(dyDist, i/2*const.value('Bohr radius'), k/2-1/2)
        
    if sparse:
        H = sp.lil_matrix((k**2,k**2))
    else:
        H = np.matrix(np.zeros((k**2,k**2)))
    
    for i in range(k**2):
        dxr = dxDist(i%k+1)-dxDist(i%k)
        dxl = dxDist(i%k)-dxDist(i%k-1)
        dxg = (dxl+dxr)/2
        dyr = dyDist(i//k+1)-dyDist(i//k)
        dyl = dyDist(i//k)-dyDist(i//k-1)
        dyg = (dyl+dyr)/2
        if i%k!=k-1:
            H[i,i+1]-=1/dxr/dxg
        if i%k!=0:
            H[i,i-1]-=1/dxl/dxg
        if i+k<k**2:
            H[i,i+k]-=1/dyr/dyg
        if i-k>=0:
            H[i,i-k]-=1/dyl/dyg
        
        H[i,i]+=1/dxl/dxg+1/dxr/dxg+1/dyl/dyg+1/dyr/dyg+V(dxDist(i%k),dyDist(i//k))
    
    H=H*const.hbar**2/2/const.m_e/const.eV
    
    if sparse:
        return H.tocsc()
    else:
        return H
        
def cartesian3D(V,k,i,dxDist=linear,dyDist=linear,dzDist=linear,sparse=True):
    dxDist = cumulativeDistributionGenerator(dxDist, i/2*const.value('Bohr radius'), k/2-1/2)
    dyDist = cumulativeDistributionGenerator(dyDist, i/2*const.value('Bohr radius'), k/2-1/2)
    dzDist = cumulativeDistributionGenerator(dyDist, i/2*const.value('Bohr radius'), k/2-1/2)
        
    if sparse:
        H = sp.lil_matrix((k**3,k**3))
    else:
        H = np.matrix(np.zeros((k**3,k**3)))
    
    for i in range(k**3):
        m = i%k
        n = (i//k)%k
        o = (i//k**2)
        dxr = dxDist(m+1)-dxDist(m)
        dxl = dxDist(m)-dxDist(m-1)
        dxg = (dxl+dxr)/2
        dyr = dyDist(n+1)-dyDist(n)
        dyl = dyDist(n)-dyDist(n-1)
        dyg = (dyl+dyr)/2
        dzr = dyDist(o+1)-dyDist(o)
        dzl = dyDist(o)-dyDist(o-1)
        dzg = (dzl+dzr)/2
        if m!=k-1:
            H[i,i+1]-=1/dxr/dxg
        if m!=0:
            H[i,i-1]-=1/dxl/dxg
        if n!=k-1:
            H[i,i+k]-=1/dyr/dyg
        if n!=0:
            H[i,i-k]-=1/dyl/dyg
        if o!=k-1:
            H[i,i+k**2]-=1/dzr/dzg
        if o!=0:
            H[i,i-k**2]-=1/dzl/dzg
        
            
        
        H[i,i]+=1/dxl/dxg+1/dxr/dxg+1/dyl/dyg+1/dyr/dyg+1/dzl/dzg+1/dzr/dzg +V(dxDist(m),dyDist(n),dzDist(o))
    
    H=H*const.hbar**2/2/const.m_e/const.eV
    
    if sparse:
        return H.tocsc()
    else:
        return H

def polarHamiltonianWithDistribution(V,kr,kphi,i,drDist=linear,sparse=False):
    drDist = cumulativeDistributionGenerator(drDist, i*const.value('Bohr radius'), -10**-6, kr-1)
    dphiDist = cumulativeDistributionGenerator(dphiDist, 2*math.pi, 0, kphi)
    dphi = dphiDist(1)-dphiDist(0)
    
    size = kr*kphi
    if sparse:
        H = sp.lil_matrix((size,size))
    else:
        H = np.matrix(np.zeros((size,size)))
    
    for i in range(size):
        m = i%kphi
        n = i//kr
        drl=drDist(n+1)-drDist(n)
        drr=drDist(n+2)-drDist(n+1)
        if m!=kphi-1:
            H[i,i+1]-=1/dphi**2/drDist(n)**2
        else:
            H[i,i-kphi+1]-=1/dphi**2/drDist(n)**2
        if m!=0:
            H[i,i-1]-=1/dphi**2/drDist(n)**2
        else:
            H[i,i+kphi-1]-=1/dphi**2/drDist(n)**2
        if i+kr<size:
            H[i,i+kr]-=1/drl/drr+1/drl**2+1/drl/drDist(n)
        if i+2*kr<size:
            H[i,i+2*kr]-=-1/drl/drr
        
        H[i,i]+=1/drl**2+1/drl/drDist(n)+2/dphi**2/drDist(n)**2 +V(drDist(n),dphiDist(m),radial=True)
        
    H=H*const.hbar**2/2/const.m_e/const.eV
    if sparse:
        return H.tocsc()
    else:
        return H
    

def inverse(g,b,eps=1.1102230246251565e-16):
    y=0.
    z=1.
    while g(z)<b:
        z*=2
    
    while abs(y-z)>eps*(abs(y)+abs(z)):
        x=(y+z)/2
        if g(x)<b:
            y=x
        else:
            z=x
    return (y+z)/2

def vectorToMatrix(v):
    k = int(v.size**0.5)
    M = np.matrix(np.zeros((k,k)))
    for i in range(v.size):
        M[i%k,i//k]=abs(v[i])**2
    return M
