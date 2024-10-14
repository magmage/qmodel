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

H0_Rabi_KS = num_op + 1/2 - t*sigma_x # with 1/2 in harmonic oscillator
CouplingRabi = x_op*sigma_z

def check_sigma_x(lam, sigma, x, j):
    # sigma and x must be in a precise relation after d/d xi applied on displacement rule
    if abs(lam*sigma + j + x) > 10e-4:
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
        
        # solve KS and full system for test
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
    ax.plot(sigma_space, test_F_full, 'yx', label=r'test for  $F^\lambda(\sigma,\xi)$')
    ax.plot(sigma_space, test_F_KS, 'yx', label=r'test for $F^0(\sigma,\xi)$')
    
    ## approximations: correct 1/2 in HO
    #ax.plot(sigma_space, 1-t*np.sqrt(1-np.square(sigma_space))+x**2, 'g.', label=r'$1-t\sqrt{1-\sigma^2}+\xi^2$')
    #approx = lambda sigma: 1-t*sqrt(1-sigma**2)+(x+lam*sigma/2)**2 - lam**2/4
    #ax.plot(sigma_space, list(map(approx, sigma_space)), 'rx', label='approx') #label=r'$1-t\sqrt{1-\sigma^2}+\frac{1}{2}\xi^2+\lambda\sigma\xi-\frac{\lambda^2}{4}(1-\sigma^2)$')
    #approx_MY = moreau_envelope(sigma_space, approx, eps)
    #ax.plot(sigma_space, approx_MY, 'r--', label=r'(1d) MY of approx')
    
    ax.legend()
    
plot_functionals_in_sigma()

def plot_in_lambda():
    ## test lambda behavior
    ## almost degeneracy at lam = 8 if v=j=0
    ## for v \neq 0 a higher lambda leads to a full up filling
    v = -0.1 # fixed
    j = 0 # fixed
    lam_space = np.linspace(0,15,100)
    sigma_array = []
    x_array = []
    eig_array = []
    eig_diff = []
    for lam in lam_space:
        H0 = H0_Rabi_KS + lam*CouplingRabi
        sol = EnergyFunctional(H0, [sigma_z, x_op]).solve([v,j]) # Hamiltonian, with coupling and external potential
        Psi0 = sol['gs_vector']
        sigma_array.append(sigma_z.expval(Psi0, transform_real = True))
        x_array.append(x_op.expval(Psi0, transform_real = True))
        eig_array.append(np.real(sol['eigenvalues'][:10])) # lowest eigenvalues to see possible crossings
        eig_diff.append(sol['eigenvalues'][1].real - sol['eigenvalues'][0].real)
    
    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.suptitle(f'expectation values in $\\lambda$ ($t = {t}, v = {v}, j = {j}$)')
    axs[0].plot(lam_space, sigma_array)
    axs[0].set_xlabel(r'$\lambda$')
    axs[0].set_ylabel(r'$\sigma$')
    axs[1].plot(lam_space, x_array)
    axs[1].set_xlabel(r'$\lambda$')
    axs[1].set_ylabel(r'$x$')
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(f'eigenvalues in $\\lambda$ ($t = {t}, v = {v}, j = {j}$)')
    ax.plot(lam_space, eig_array)
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(f'difference of two lowest eigenvalues in $\\lambda$ ($t = {t}, j = {j}, v = {v}$)')
    ax.plot(lam_space, eig_diff)
    ax.set_yscale('log')
    
#plot_in_lambda()

def plot_photon_filling():
    lam = 2
    v = -0.1 # fixed
    j = 0 # fixed
    Psi0 = EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [sigma_z, x_op]).solve([v, j])['gs_vector']
    
    sigma_expval = sigma_z.expval(Psi0, transform_real = True)
    x_expval = x_op.expval(Psi0, transform_real = True)
    print('sigma_z expectation value = {}'.format(sigma_expval))
    print('x expectation value = {}'.format(x_expval))
    check_sigma_x(lam, sigma_expval, x_expval, j)
    
    rho0_up = [ b.hop({'n': n, 's': +1}).expval(Psi0, transform_real = True) for n in range(oscillator_size) ]
    rho0_down = [ b.hop({'n': n, 's': -1}).expval(Psi0, transform_real = True) for n in range(oscillator_size) ]

    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.suptitle(f'photon filling ($\\lambda = {lam}, t = {t}, j = {j}, v = {v}$)')
    ymax = max(max(rho0_up), max(rho0_down)) * 1.2
    print('sum of rho0 = {}'.format(sum(rho0_up)+sum(rho0_down)))
    
    axs[0].set_title('spin up', size=10)
    axs[0].bar(range(oscillator_size), rho0_up)
    axs[0].set_ylim(0, ymax)

    axs[1].set_title('spin down', size=10)
    axs[1].bar(range(oscillator_size), rho0_down)
    axs[1].set_ylim(0, ymax)
    
#plot_photon_filling()

def plot_functionals_in_sigma_for_paper():
    sigma_space = np.linspace(-1,1,301)
    #xi_space = np.linspace(-1,1,9)
    lam_space = np.linspace(0,2,5)
    #colors = plt.cm.twilight(np.linspace(0,1,len(xi_space)))
    colors = plt.cm.tab10(np.linspace(0,1,len(lam_space)))
    
    lam = 0
    x = 0
    eps = 0.1
    
    #E = EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [sigma_z, x_op])
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    
    #for i,x in enumerate(xi_space):
    for i,lam in enumerate(lam_space):
        E = EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [sigma_z, x_op])
        F = []
        
        for sigma in sigma_space:
            LT = E.legendre_transform([sigma, x])
            check_sigma_x(lam, sigma, x, LT['pot'][1])
            F.append(LT['F'])
        
        #ax.plot(sigma_space, F, color=colors[i], linewidth=(1 if x != 0 else 2), label=r'$\xi={:.2f}$'.format(x))
        ax.plot(sigma_space, F, color=colors[i], label=r'$\lambda={:.2f}$'.format(lam))
    
    ax.legend()
    ax.set_xlabel(r'$\sigma$')
    #ax.set_ylabel(r'$F_\mathrm{LL}(\sigma,\xi)$')
    ax.set_ylabel(r'$F_\mathrm{LL}(\sigma,0)$')

#plot_functionals_in_sigma_for_paper()
plt.show()