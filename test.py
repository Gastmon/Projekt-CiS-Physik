import hamiltonian as ham
import cis
import plotter
import scipy.constants as const
import scipy.sparse.linalg as splinalg
import numpy as np
import time
import sys

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
    
def check(k=60,i=32,sparse=False,numOfEV=6,plot=False):
    print('check(k='+str(k)+', i='+str(i)+', sparse='+str(sparse)+', numOfEV='+str(numOfEV)+')',file=sys.stderr)
    H=ham.hamiltonianWithDistribution(ham.coloumbPotential,k,i,sparse=sparse)
    
    if sparse:
        w,v=splinalg.eigsh(H,k=numOfEV,which='SA')
    else:
        w,v=np.linalg.eigh(H)
    
    for n in range(numOfEV):
        if plot:
            plotForOneValue(n,w[n],v[:,n])
        else:
            plotForOneValue(n,w[n])
    
def plotForOneValue(n,E_n,v_n=None):
    print('n: '+str(n)+'\tE_n(J): '+str(E_n*const.eV)+'\tE_n(eV): '+str(E_n),file=sys.stderr)
    if v_n is not None:
        plotter.plotMatrix(ham.vectorToMatrix(v_n))
    
def testQR(k=16,i=16,iterations=500):
    print('QReigenvalues(k='+str(k)+', i='+str(i)+', iterations='+str(iterations)+')')#,file=sys.stderr)
    H=ham.hamiltonianWithDistribution(ham.coloumbPotential,k,i,sparse=False)
    t0=time.time()
    w,v=cis.QReigenvalues(H,iterations=iterations)
    #sortedE = np.sort(E).T
    #sortedv = [v[:,list(E).index(x)] for x in sortedE[:6]]
    #print(sortedE[:6,0])
    t1=time.time()
    sort_index = np.argsort(w)
    for n in range(12):
        plotForOneValue(n, float((w.T)[sort_index[0,n]]), v[:,sort_index[0,n]])
    print('Time needed: '+str(t1-t0))#,file=sys.stderr)
    
def testSparseHamiltonian():
    for k in range(1,7):
        for i in range(7):
            H = ham.hamiltonian(ham.coloumbPotential, 2**k, 2**i*const.value('Bohr radius')/2**k, 2**i*const.value('Bohr radius')/2**k)
            Hsparse = ham.hamiltonian(ham.coloumbPotential, 2**k, 2**i*const.value('Bohr radius')/2**k, 2**i*const.value('Bohr radius')/2**k, sparse=True)
            if not (H==Hsparse).all():
                print('Error for k = '+str(k)+', i = '+str(i))

def testJacobi(k,i,iterations):
    print('cyclicJacobiAlgorithm(k='+str(k)+', i='+str(i)+', iterations='+str(iterations)+')')#,file=sys.stderr)
    H=ham.hamiltonianWithDistribution(ham.coloumbPotential,k,i,sparse=False)
    t0=time.time()
    w,v=cis.cyclicJacobiAlgorithm(H,iterations)
    t1=time.time()
    w = sorted(w)
    
    for n in range(12):
        plotForOneValue(n,float(w[n]),v[:,n])
    print('Time needed: '+str(t1-t0))#,file=sys.stderr)
    
def testTridiagonalBisection(k=60,i=32,m=0):
    m=k#*int(k**0.5)
    print('bisection(k='+str(k)+', i='+str(i)+', iterations='+str(m)+')',file=sys.stderr)
    H=ham.hamiltonianWithDistribution(ham.coloumbPotential,k,i,sparse=True)
    a,b = cis.lanczos(H,m)
    eps=1.0
    while(1+eps!=1):
        eps/=2#eps=1.1102230246251565e-16
    c=cis.signCount(a,b)
    E=cis.countBisection(c,-10**4,0,eps,len(a),0)[0]
    plotForOneValue(0,E)

def testTriBiInv(k=60,i=32,m=0,numOfEV=1,plot=False,dist=ham.linear):
    H=ham.hamiltonianWithDistribution(ham.coloumbPotential,k,i,dist,dist,sparse=True)
    
    m=k
    a,b = cis.lanczos(H,m)
    eps=1.0
    while(1+eps!=1):
        eps/=2#eps=1.1102230246251565e-16
    c=cis.signCount(a,b)
    E=cis.countBisection(c,-10**4,0,eps,len(a),0)[0]
    plotForOneValue(0,E)
    E,v=cis.inversePowerMethod(H,E,15)
    plotForOneValue(0,E)
    
    
def testDistHamiltonian(k=16,i=16,sparse=True,numOfEV=6,plot=False,dist=ham.linear):
    print('testDistHamiltonian(k='+str(k)+', i='+str(i)+', sparse='+str(sparse)+', numOfEV='+str(numOfEV)+')')#,file=sys.stderr)
    H=ham.hamiltonianWithDistribution(ham.coloumbPotential,k,i,dist,dist,sparse=sparse)
    
    if sparse:
        w,v=splinalg.eigsh(H,k=numOfEV,which='SA')
    else:
        w,v=np.linalg.eigh(H)
    
    for n in range(numOfEV):
        if plot:
            plotForOneValue(n,w[n],v[:,n])
        else:
            plotForOneValue(n,w[n])

def testCartesian3D(k=16,i=16,sparse=True,numOfEV=6,plot=False,dist=ham.linear):
    print('testCartesian3D(k='+str(k)+', i='+str(i)+', sparse='+str(sparse)+', numOfEV='+str(numOfEV)+')')#,file=sys.stderr)
    H=ham.cartesian3D(ham.coloumbPotential,k,i,dist,dist,dist,sparse=sparse)
    
    if sparse:
        w,v=splinalg.eigsh(H,k=numOfEV,which='SA')
        #eps=1.1102230246251565e-16
        #a,b = cis.lanczos(H,2400)
        #c=cis.signCount(a,b)
        #w=[cis.countBisection(c,-10**4,0,eps,len(a),0)[0]]
    else:
        w,v=np.linalg.eigh(H)
    
    for n in range(numOfEV):
        if plot:
            plotForOneValue(n,w[n],v[:,n])
        else:
            plotForOneValue(n,w[n])   

def timeQR(k):
    H=ham.hamiltonianWithDistribution(ham.coloumbPotential,k,16,sparse=False)
    print('QReigenvalues(k='+str(k)+', i='+str(16)+', iterations='+str(20)+')')#,file=sys.stderr)
    t0=time.time()
    w,v=cis.QReigenvalues(H,20)
    t1=time.time()
    print('Time needed: '+str(t1-t0))#,file=sys.stderr)
    plotForOneValue(0,min(w[0,0]))

if __name__ == '__main__':
    k=185
    print('k='+str(k))
    H=ham.hamiltonianWithDistribution(ham.coloumbPotential,k,16,sparse=False)
    #testQR(k,16,100*k)
    #testJacobi(k,16,5)
    #t1=time.time()
    #check(256,16,True,1,plot=False)
    t0=time.time()
    #print('Time needed: '+str(t0-t1))#,file=sys.stderr)
    #testQR(16,32)
    #testTridiagonalBisection(k,16)
    #check(k,16,True,1,plot=False)
    #print('testJacobi(10,16) with 16000 iterations')
    #check(256,16,True)
    #print('testQR(10,16)')
    #testDistHamiltonian(k=64,i=16,dist=ham.cubicOrderSinusHyperbolicus)
    #testTriBiInv(400,16,dist=ham.linear)
    #testCartesian3D(62,numOfEV=1,dist=ham.cubic)
    #check(16,16)
    #check(k,16,plot=False,numOfEV=12)
    t1=time.time()
    print('Time needed: '+str(t1-t0),file=sys.stderr)
    
    
    
