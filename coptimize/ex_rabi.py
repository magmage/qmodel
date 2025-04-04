'''
Constrained Search in Imaginary Time
Quantum Rabi model example
cf. https://arxiv.org/abs/2411.15256 for model and code details
'''

from qmodel import NumberBasis, SpinBasis, OperatorList, timer
from coptimize import ConstraintOptimization
import numpy as np
import matplotlib.pyplot as plt 

oscillator_size = 10
b_oscillator = NumberBasis(oscillator_size)
b_spin = SpinBasis()
b = b_oscillator.tensor(b_spin)

num_op = b_oscillator.diag().extend(b) # number operator for oscillator
x_op = b_oscillator.x_operator().extend(b)
sigma_z = b_spin.sigma_z().extend(b)
sigma_x = b_spin.sigma_x().extend(b)

t = 1
lam = 1

A = 1/2*(num_op + 1/2) - t*sigma_x + lam*x_op*sigma_z
B = OperatorList(b, [sigma_z])

co = ConstraintOptimization(A, B)

@timer
def plot_functionals_in_sigma():
    sigma_space = np.linspace(-1,1,11)
    lam_space = np.linspace(0,2,5)
    colors = plt.cm.tab10(np.linspace(0,1,len(lam_space)))
    
    fig, ax = plt.subplots(ncols=1, nrows=1)    
    for i,lam in enumerate(lam_space):
        F = []
        for sigma in sigma_space:
            res = co.run([sigma])
            F.append(res['min'])
        
        ax.plot(sigma_space, F, color=colors[i], label=r'$\lambda={:.2f}$'.format(lam))
    
    ax.legend()
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$F_\mathrm{LL}(\sigma,0)$')

plot_functionals_in_sigma()
plt.show()