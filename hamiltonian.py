import numpy as np

def hamiltonian(V,n):
    H=np.matrix(np.zeros((n**2,n**2)))
    for i in range(n**2):
        if i%n!=n-1:
            H[i,i+1]-=1
        if i%n!=0:
            H[i,i-1]-=1
        if i+n<n**2:
            H[i,i+n]-=1
        if i-n>=0:
            H[i,i-n]-=1
        H[i,i]+=4+V(i,i)
    return H
        

def nullPotential(i,j):
    return 0
