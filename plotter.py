import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def plotWavefunction(v):
    k = v.size**0.5
    x = []
    y = []
    values = []
    for i in range(v.size):
        x.append(i%k)
        y.append(i//k)
        values.append(float(v[i]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, values)
    plt.show()

def plotMatrix(M):
    k = M.shape[0]
    x = np.linspace(0,k-1,k)
    X,Y = np.meshgrid(x,x)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.contour3D(X,Y,np.array(M),50,cmap='binary')
    #ax.plot_wireframe(X,Y,np.array(M))
    ax.plot_surface(X, Y, np.array(M), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()
