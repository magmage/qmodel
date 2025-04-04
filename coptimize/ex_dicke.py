'''
Constrained Search in Imaginary Time
Dicke model example
cf. https://arxiv.org/abs/2411.15256 and https://arxiv.org/abs/2409.13767 for model and code details
'''

from qmodel import NumberBasis, SpinBasis, OperatorList, timer
from coptimize import ConstraintOptimization
import numpy as np
import matplotlib.pyplot as plt 

# Dicke model example for N=2, M=1
# just v * sigma coupling

## needs very low accuracy :(
## why do sign flips never occur again? -> positive gs

## 1d cut

oscillator_size = 4
b_oscillator = NumberBasis(oscillator_size)
# define qubits separately so also operators can act separately on them
b_spin1 = SpinBasis('s1')
b_spin2 = SpinBasis('s2')
b = b_oscillator.tensor(b_spin1.tensor(b_spin2))

num_op = b_oscillator.diag().extend(b) # number operator for oscillator
x_op = b_oscillator.x_operator().extend(b)
sigma_z1 = b_spin1.sigma_z().extend(b) # also possible as vec
sigma_z2 = b_spin2.sigma_z().extend(b)
sigma_x1 = b_spin1.sigma_x().extend(b)
sigma_x2 = b_spin2.sigma_x().extend(b)

t = 1
lam = 1
A = 2*(num_op + 1/2) - t*(sigma_x1 + sigma_x2) + lam*x_op*(sigma_z1 + sigma_z2)
B = OperatorList(b, [sigma_z1, sigma_z2] )

co = ConstraintOptimization(A, B)

@timer
def DickeRun():    
    resol = 10 # resolution
    F = np.zeros((resol,resol))
    linsp = np.linspace(-1,1,resol)
    for ix in range(resol):
        for iy in range(resol):
            print(ix, iy)
            x = linsp[ix]
            y = linsp[iy]
            sigma = [x,y]
            # run
            res = co.run(sigma)
            F[iy,ix] = res['min'] # rows,cols
    
    plt.imshow(F, cmap='managua_r', interpolation='none', extent=[-1,1,-1,1])
    plt.colorbar()

#DickeRun()

@timer
def DickeRun1D():    
    resol = 50 # resolution
    F = np.zeros((resol,))
    linsp = np.linspace(-1,1,resol)
    for i in range(resol):
        print(i)
        x = linsp[i]
        y = 0
        sigma = [x,y]
        # run
        res = co.run(sigma)
        F[i] = res['min']
    
    plt.plot(F)

DickeRun1D() # seems to work very well

plt.show()