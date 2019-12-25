import numpy as np
import scipy.sparse as sp


def powerMethod(A, iterations=300):
    (m, n) = A.shape
    v = np.matrix(np.random.rand(n, 1)+np.random.rand(n, 1)*1j)
    E = []
    for k in range(iterations-1):
        v = A * v
        v = v / np.linalg.norm(v)
        #E.append(float(v.T * A * v))
    E1=complex(v.H * A * v)
    v = A * v
    v = v / np.linalg.norm(v)
    E2=complex(v.H * A * v)
    print('Absolute Error: '+str(E2-E1))
    print('Relative Error: '+str(abs(E2-E1)/abs(E2)))
    return (E2,v)


def QReigenvalues(A, iterations=20, qr=np.linalg.qr, p=lambda x:x):
    (m, n) = A.shape
    Q = np.matrix(np.identity(n))
    for k in range(iterations):
        Q_k,R = qr(p(A))
        A = R*Q_k
        Q = Q*Q_k#Q*D*Q.T=A
    return A.diagonal(),Q

def QRhouseholder(A):
    (m,n) = A.shape
    Q = np.matrix(np.identity(m))
    R = np.matrix(np.zeros((m,n)))
    I = np.matrix(np.identity(m))
    t = min(m-1,n)
    
    for k in range(t):
        x=A[k:,k]
        alpha=np.sign(x[0,0])*np.linalg.norm(x)
        if alpha==0:
            print('Error in QRhouseholder: Pivot is 0')
        u=x-alpha*I[k:,k]
        v=u/np.linalg.norm(u)
        Q_k=I.copy()
        Q_k[k:,k:]=I[k:,k:]-2*v*v.H
        A=Q_k*A
        Q=Q*Q_k.T
    
    return Q,A

def originalJacobiAlgorithm(A,iterations):
    (m,n) = A.shape
    V = sp.identity(m).tocsc()
    for k in range(iterations):
        A,V = jacobiStepMatrix(A,V)
    return A.diagonal().T,V

def cyclicJacobiAlgorithm(A,iterations):
    (m,n) = A.shape
    V = np.matrix(np.identity(m))
    for k in range(iterations):
        for p in range(m-1):
            for q in range(p+1,m):
                A,V = jacobiStepIndex(A,V,p,q)
    return A.diagonal().T,V

def jacobiStepIndex(A,V,p,q):
    (m,n) = A.shape
    if sp.isspmatrix(A) and not sp.isspmatrix_lil(A):
        a=A.tolil()
    else:
        a=A.copy()
    v=V.copy()
    theta = (A[q,q]-A[p,p])/2/A[p,q]
    t = np.sign(theta)/(np.abs(theta)+(theta**2+1)**0.5)
    c = 1/(t**2+1)**0.5
    s = t*c
    tau = s/(1+c)
    
    a[p,p]=A[p,p]-t*A[p,q]
    a[q,q]=A[q,q]+t*A[p,q]
    a[p,q]=(c**2-s**2)*A[p,q]+s*c*(A[p,p]-A[q,q])
    a[q,p]=a[p,q]
    
    for r in range(m):
        if r!=p and r!=q:
            arp = A[r,p]
            arq = A[r,q]
            a[r,p] = A[r,p]-s*(A[r,q]+tau*A[r,p])
            a[r,q] = A[r,q]+s*(A[r,p]-tau*A[r,q])
            
            a[p,r] = a[r,p]
            a[q,r] = a[r,q]
        
        v[r,p] = V[r,p]-s*(V[r,q]+tau*V[r,p])
        v[r,q] = V[r,q]+s*(V[r,p]-tau*V[r,q])
    
    return a,v

def jacobiStepMatrix(A,V,p=None,q=None):
    (m,n) = A.shape
    if p==None or q==None:
        p,q = getMaxPivot(A)
    theta = (A[q,q]-A[p,p])/2/A[p,q]
    t = np.sign(theta)/(np.abs(theta)+(theta**2+1)**0.5)
    #print('theta: '+str(theta)+'t: '+str(t))
    c = 1/(t**2+1)**0.5
    s = t*c
    #print('c: '+str(c)+'\t,s: '+str(s))
    P = sp.identity(m).tolil()
    P[p,p] = c
    P[p,q] = s
    P[q,p] =-s
    P[q,q] = c
    P = P.tocsc()
    #print(P.toarray())
    A = P.T*A*P
    return A,V*P
    
def getMaxPivot(a):
    (m,n) = a.shape
    A=abs(a)
    p = 0
    q = 0
    for k in range(m):
        A[p,q]=0
    while(True):
        flatIndex = A.argmax()
        p = flatIndex//n
        q = flatIndex%n
        if a[p,p] != a[q,q]:
            #print('p: '+str(p)+'\t,q: '+str(q))
            return p, q
        else:
            A[p,q]=0
            A[q,p]=0
            
def bisection(f,y,z,eps):
    if f(y)*f(z)>0:
        print('error')
        return
    while(abs(y-z)>eps*(abs(y)+abs(z))):
        x=(y+z)/2
        if f(x)*f(y)<0:
            z=x
        else:
            y=x
    return (y+z)/2
    
def countBisection(c,y,z,eps,n,k):
    while(abs(y-z)>eps*(abs(y)+abs(z))):
        x=(y+z)/2
        temp=c(x)
        #print('y: '+str(y)+', \tz: '+str(z)+',   \tx: '+str(x)+', \tc(x): '+str(temp))
        if temp>k:
            z=x
        else:
            y=x
    return y,z
    
def polynomialGenerator(a,b,n=None):
    if n==None:
        n=len(a)
    def fun(x):
        p=[1, a[0]-x]
        for r in range(2,n+1):
            p.append((a[r-1]-x)*p[r-1]-b[r-2]**2*p[r-2])
        return p[-1]
    return fun
    
def signCount(a,b):
    def fun(x):
        p=[1, a[0]-x]
        count=0
        for r in range(2,len(a)+1):
            p.append((a[r-1]-x)*p[r-1]-b[r-2]**2*p[r-2])
        for r in range(1,len(a)+1):
            if np.sign(p[r])==0:
                count+=1
            elif np.sign(p[r-1])!=0 and np.sign(p[r])!=np.sign(p[r-1]):
                count+=1
        #print(p)
        return count
    return fun
