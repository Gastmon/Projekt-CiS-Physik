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
    w,v=cis.QReigenvalues(H,iterations=1000, qr=cis.QRhouseholder)
    #sortedE = np.sort(E).T
    #sortedv = [v[:,list(E).index(x)] for x in sortedE[:6]]
    #print(sortedE[:6,0])
    sort_index = np.argsort(w)
    for n in range(6):
        plotForOneValue(n, float((w.T)[sort_index[0,n]]), v[:,sort_index[0,n]])
    
def testSparseHamiltonian():
    for k in range(1,7):
        for i in range(7):
            H = ham.hamiltonian(ham.coloumbPotential, 2**k, 2**i*const.value('Bohr radius')/2**k, 2**i*const.value('Bohr radius')/2**k)
            Hsparse = ham.hamiltonian(ham.coloumbPotential, 2**k, 2**i*const.value('Bohr radius')/2**k, 2**i*const.value('Bohr radius')/2**k, sparse=True)
            if not (H==Hsparse).all():
                print('Error for k = '+str(k)+', i = '+str(i))

def testJacobi(k,i):
    H=ham.hamiltonian(ham.coloumbPotential,k,i*const.value('Bohr radius')/k,i*const.value('Bohr radius')/k)
    w,v=cis.originalJacobiAlgorithm(H,16000)
    w = sorted(w)
    
    for n in range(6):
        plotForOneValue(n,float(w[n]),v[:,n])
    
def testTridiagonalBisection(k=60,i=32):
    H=ham.hamiltonian(ham.coloumbPotential,k,i*const.value('Bohr radius')/k,i*const.value('Bohr radius')/k,True)
    a=H.diagonal()
    b=H.diagonal(k=1)
    eps=1.0
    while(1+eps!=1):
        eps/=2#eps=1.1102230246251565e-16
    c=cis.signCount(a,b)
    v=cis.countBisection(c,-10**0,0,eps,len(a),0)
    print(v)
    

if __name__ == '__main__':
    t0=time.time()
    #testQR(16,32)
    check(256,16,True,1)
    testTridiagonalBisection(256,16)
    #testJacobi(10,16)
    #print('testJacobi(10,16) with 16000 iterations')
    #check(20,16)
    #testQR(10,16)
    #print('testQR(10,16)')
    t1=time.time()
    #check(10,16)
    print('Time needed: '+str(t1-t0))
    
