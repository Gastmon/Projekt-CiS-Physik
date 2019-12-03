import hamiltonian as ham
import cis
import plotter
import scipy.constants as const
import scipy.sparse.linalg as splinalg
import numpy as np
import time

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
    
def check(k=60,i=32,sparse=False,numOfEV=6):
    H=ham.hamiltonian(ham.coloumbPotential,k,i*const.value('Bohr radius')/k,i*const.value('Bohr radius')/k,sparse)
    
    if sparse:
        w,v=splinalg.eigsh(H,k=numOfEV,which='SA')
    else:
        w,v=np.linalg.eigh(H)
    
    for n in range(numOfEV):
        plotForOneValue(n,w[n],v[:,n])
    
def plotForOneValue(n,E_n,v_n):
    print('n: '+str(n)+'\tE_n(J): '+str(E_n)+'\tE_n(eV): '+str(E_n/const.eV))
    #plotter.plotMatrix(ham.vectorToMatrix(v_n))
    
def testQR(k=60,i=32):
    H=ham.hamiltonian(ham.coloumbPotential,k,i*const.value('Bohr radius')/k,i*const.value('Bohr radius')/k)
    E,v=cis.QReigenvalues(H,iterations=10000)#, qr=cis.QRhouseholder)
    #sortedE = np.sort(E).T
    #sortedv = [v[:,list(E).index(x)] for x in sortedE[:6]]
    #print(sortedE[:6,0])
    sort_index = np.argsort(E)
    for n in range(6):
        plotForOneValue(n, float((E.T)[sort_index[0,n]]), v[:,sort_index[0,n]])
    
def testSparseHamiltonian():
    for k in range(1,7):
        for i in range(7):
            H = ham.hamiltonian(ham.coloumbPotential, 2**k, 2**i*const.value('Bohr radius')/2**k, 2**i*const.value('Bohr radius')/2**k)
            Hsparse = ham.hamiltonian(ham.coloumbPotential, 2**k, 2**i*const.value('Bohr radius')/2**k, 2**i*const.value('Bohr radius')/2**k, sparse=True)
            if not (H==Hsparse).all():
                print('Error for k = '+str(k)+', i = '+str(i))

if __name__ == '__main__':
    t0=time.time()
    #testQR(16,32)
    check(512,64,True)
    t1=time.time()
    print('Time needed: '+str(t1-t0))
