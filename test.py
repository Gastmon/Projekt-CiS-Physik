import hamiltonian as ham
import cis
import plotter
import scipy.constants as const

def scatter(k):
    H=ham.hamiltonian(ham.coloumbPotential,k)
    (E,v)=cis.powerMethod(H)
    print(E)
    plotter.plotWavefunction(v)
    plotter.plotWavefunction(H.diagonal().T)
    return v

def contour(k,i=1):
    H=ham.hamiltonian(ham.coloumbPotential,k,i*const.value('Bohr radius')/k,i*const.value('Bohr radius')/k)
    (E,v)=cis.powerMethod(H)
    print(E)
    print(E/const.eV)
    plotter.plotMatrix(ham.vectorToMatrix(v))
    #plotter.plotMatrix(ham.vectorToMatrix(H.diagonal().T))
    return v
    
if __name__ == '__main__':
    contour(40,2)
    contour(40,4)
    contour(40,8)
    contour(40,16)
    contour(40,32)
