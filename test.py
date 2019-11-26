import hamiltonian as ham
import cis
import plotter
import scipy.constants as const
import numpy as np

def scatter(k):
    H=ham.hamiltonian(ham.coloumbPotential,k)
    (E,v)=cis.powerMethod(H)
    print(E)
    plotter.plotWavefunction(v)
    plotter.plotWavefunction(H.diagonal().T)
    return v

def contour(k,i=1):
    H=ham.hamiltonian(ham.coloumbPotential,k,i*const.value('Bohr radius')/k,i*const.value('Bohr radius')/k)
    (E,v)=cis.powerMethod(H,300)
    print(E)
    print(E/const.eV)
    plotter.plotMatrix(ham.vectorToMatrix(v))
    #plotter.plotMatrix(ham.vectorToMatrix(H.diagonal().T))
    return v
    
def check(k=60,i=32):
    H=ham.hamiltonian(ham.coloumbPotential,k,i*const.value('Bohr radius')/k,i*const.value('Bohr radius')/k)
    w,v=np.linalg.eigh(H)
    for n in range(6):
        plotForOneValue(n,w,v)
    return v
    
def plotForOneValue(n,w,v):
    E_n=w[n]
    v_n=v[:,n]
    print('n: '+str(n)+'\tE_n(J): '+str(E_n)+'\tE_n(eV): '+str(E_n/const.eV))
    plotter.plotMatrix(ham.vectorToMatrix(v_n))
    
def testQR(k=60,i=32):
    H=ham.hamiltonian(ham.coloumbPotential,k,i*const.value('Bohr radius')/k,i*const.value('Bohr radius')/k)
    E=cis.QReigenvalues(H, qr=cis.QRhouseholder)
    print((np.sort(E).T)[:8,0])

if __name__ == '__main__':
    testQR(16,16)
