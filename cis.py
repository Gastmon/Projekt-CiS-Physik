import numpy as np


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
    #(m, n) = A.shape
    #Q = np.matrix(np.identity(n))
    for k in range(iterations):
        Q_k,R = qr(p(A))
        A = R*Q_k
        #Q = Q*Q_k
    return A.diagonal()

# function [Q,R]=B5_aufg3ii(A)
# [m,n]=size(A);
# Q=zeros(m,n);
# R=zeros(n);

# for k=1:n
    # R(k,k)=norm(A(:,k));
    # Q(:,k)=A(:,k)/R(k,k);
    # for i=k+1:n
        # R(k,i)=Q(:,k)'*A(:,i);
        # A(:,i)=A(:,i)-R(k,i)*Q(:,k);
    # end
# end

def modQR(A):
    (m,n) = A.shape
    Q = np.matrix(np.zeros((m,n)))
    R = np.matrix(np.zeros((n,n)))
    
    for k in range(n):
        R[k,k]=np.linalg.norm(A[:,k])
        Q[:,k]=A[:,k]/R[k,k]
        for i in range(k+1,n):
            R[k,i]=Q[:,k].T*A[:,i]
            A[:,i]=A[:,i]-R[k,i]*Q[:,k]
            
    print(Q)
    print('\n')
    print(R)
    return (Q,R)
    
# function [Q,R]=B5_aufg3i(A)
# [m,n]=size(A);
# Q=zeros(m,n);
# R=zeros(n);

# for k=1:n
    # R(1:k-1,k)=Q(1:m,1:k-1)'*A(1:m,k);
    # q_aux=A(1:m,k)-Q(1:m,1:k-1)*R(1:k-1,k);
    # R(k,k)=norm(q_aux);
    # Q(1:m,k)=q_aux/R(k,k);
# end

def QR(A):
    (m,n) = A.shape
    Q = np.matrix(np.zeros((m,n)))
    R = np.matrix(np.zeros((n,n)))

    for k in range(n):
        R[0:k-1,k]=Q[0:m,0:k-1].T*A[0:m,k]
        q_aux=A[0:m,k]-Q[0:m,0:k-1]*R[0:k-1,k]
        R[k,k]=np.linalg.norm(q_aux)
        Q[0:m,k]=q_aux/R[k,k]
        
    #print(Q)
    #print('\n')
    #print(R)
    return (Q,R)
