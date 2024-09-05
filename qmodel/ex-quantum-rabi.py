'''
Created on 10.08.2024

@author: mage

quantum Rabi model example
'''
from qmodel import *
from dft import *
import matplotlib.pyplot as plt
import numpy as np

oscillator_size = 20
b_oscillator = NumberBasis(oscillator_size)
b_spin = SpinBasis()
b = b_oscillator.tensor(b_spin)

num_op = b_oscillator.diag().extend(b) # number operator for oscillator
x_op = b_oscillator.x_operator().extend(b)
sigma_z = b_spin.sigma_z().extend(b)
sigma_x = b_spin.sigma_x().extend(b)

t = 1

H0_Rabi_KS = 2*(num_op + 1/2) - t*sigma_x
CouplingRabi = x_op*sigma_z

def check_sigma_x(lam, sigma, x, j):
    # sigma and x must be in a precise relation after d/d xi applied on displacement rule
    if abs(lam*sigma + j + 2*x) > 10e-4:
        print(f'sigma--xi check: FAIL at lam={lam}, sigma={sigma}, xi={x}, j={j}! maybe increase oscillator_size value')

def plot_functionals_in_sigma():
    sigma_space = np.linspace(-1,1,51)
    
    lam = 1
    x = 0.5
    eps = 0.1
    
    E_full = EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [sigma_z, x_op])
    E_KS = EnergyFunctional(H0_Rabi_KS, [sigma_z, x_op])
    F_full = []
    F_full_eps = []
    F_KS = []
    E_xc = []
    test_F_full = []
    test_F_KS = []
    F_approx = []
    v_full_eps = []
    v_full_eps_prox = []
    
    for sigma in sigma_space:
        LT_full = E_full.legendre_transform([sigma, x])
        LT_full_eps = E_full.legendre_transform([sigma, x], epsMY=eps)
        LT_KS = E_KS.legendre_transform([sigma, x])
        
        check_sigma_x(lam, sigma, x, LT_full['pot'][1])
        check_sigma_x(0, sigma, x, LT_KS['pot'][1])
        
        F_full.append(LT_full['F'])
        F_full_eps.append(LT_full_eps['F'])
        v_full_eps.append(LT_full_eps['pot'][0])
        v_full_eps_prox.append(-1/eps * (sigma - E_full.prox([sigma, x], eps)[0])) # Th. 9 in paper
        F_KS.append(LT_KS['F'])
        E_xc.append(F_full[-1] - F_KS[-1])
        
        # solve KS and full system
        sol_full = E_full.solve(LT_full['pot'])
        sol_KS = E_KS.solve(LT_KS['pot'])
        
        test_F_full.append(sol_full['gs_energy'] - np.dot(LT_full['pot'], [sigma, x]))
        test_F_KS.append(sol_KS['gs_energy'] - np.dot(LT_KS['pot'], [sigma, x]))
        
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(r'Universal functional in $\sigma$ at $\xi$={}'.format(x))
    ax.plot(sigma_space, F_full, 'b', label=r'$F^\lambda(\sigma,\xi)$, $\lambda={}$'.format(lam))
    ax.plot(sigma_space, F_full_eps, 'b--', label=r'$F_\varepsilon^\lambda(\sigma,\xi)$, $\lambda={}$, $\varepsilon={}$'.format(lam, eps))
    ax.plot(sigma_space, v_full_eps, 'k--', label=r'$v$ from $-\nabla F_\varepsilon^\lambda(\sigma,\xi)$'.format(lam, eps))
    ax.plot(sigma_space, -np.gradient(F_full_eps, sigma_space[1]-sigma_space[0]), 'kx', label=r'$v$ from numeric differentiation')
    ax.plot(sigma_space, v_full_eps_prox, 'k.', label=r'$v$ from proximal map')
    ax.plot(sigma_space, F_KS, 'g', label=r'$F^0(\sigma,\xi)$')
    ax.plot(sigma_space, 1-t*np.sqrt(1-np.square(sigma_space))+x**2, 'g.', label=r'$1-t\sqrt{1-\sigma^2}+\xi^2$')
    
    approx = lambda sigma: 1-t*sqrt(1-sigma**2)+(x+lam*sigma/2)**2 - lam**2/4
    ax.plot(sigma_space, list(map(approx, sigma_space)), 'rx', label='approx') #label=r'$1-t\sqrt{1-\sigma^2}+\frac{1}{2}\xi^2+\lambda\sigma\xi-\frac{\lambda^2}{4}(1-\sigma^2)$')
    approx_MY = moreau_envelope(sigma_space, approx, eps)
    ax.plot(sigma_space, approx_MY, 'r--', label=r'(1d) MY of approx')
    
    ax.legend()
    
    plt.show()
    
plot_functionals_in_sigma()