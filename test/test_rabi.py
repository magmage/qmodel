'''
Created on 14.02.2024
@author: mage
'''
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from qmodel import oscillator, spin, func

t = 1
lam = 1
j = 0
v = 0.2
oscillator_size = 10

## Rabi model in 2nd quantization basis
OB = oscillator.Basis(oscillator_size)
SB = spin.Basis()
B = OB.tensor(SB)

XOperator = ((oscillator.Creator(OB) + oscillator.Annihilator(OB)) / sqrt(2)).extend(B)
SigmaX = spin.SigmaX(SB).extend(B)
SigmaZ = spin.SigmaZ(SB).extend(B)
CouplingRabi = SigmaZ*XOperator
CouplingJC = spin.SigmaMinus(SB).extend(B)*oscillator.Creator(OB).extend(B) + spin.SigmaPlus(SB).extend(B)*oscillator.Annihilator(OB).extend(B)
H0_KS = -t*SigmaX + oscillator.Hamiltonian(OB).extend(B) # Hamiltonian, just kinetic
H0_full = H0_KS + lam*CouplingRabi # Hamiltonian, with coupling
H = H0_full + j*XOperator + v*SigmaZ # Hamiltonian, with external potential and current

def check_sigma_x(sigma, x):
    # sigma and x must be in a precise relation after Eq. (33) of draft
    print('sigma--x check: {}'.format('OK' if (lam*sigma + j + x < 10e-14) else 'FAIL! increase oscillator_size value'))

def plot_photon_filling():
    Psi0 = func.EnergyFunctional(H).solve()['gs_vector']
    
    sigma_expval = SigmaZ.expval(Psi0, check_real = True)
    x_expval = XOperator.expval(Psi0, check_real = True)
    print('sigma_z expectation value = {}'.format(sigma_expval))
    print('x expectation value = {}'.format(x_expval))
    check_sigma_x(sigma_expval, x_expval)
    
    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.suptitle(f'photon filling ($t = {t}, j = {j}, v = {v}$)')
    sub_titles = ['spin up', 'spin down']
    rho0 = [Psi0.filter_abssq({'spin_s': s}) for s in (-1/2,+1/2)] # spin up/down
    ymax = max([max(rho0[i]) for i in (0,1)]) * 1.2
    for i in (0,1):
        axs[i].set_title(sub_titles[i], size=10)
        axs[i].bar(range(0,oscillator_size), rho0[i])
        axs[i].set_ylim(0, ymax)
        
    plt.show()

def plot_expectation_values(): # for different external potentials
    E = func.EnergyFunctional(H0_full, [SigmaZ, XOperator])

    v_space = np.linspace(-2,2,50)
    j_space = np.linspace(-2,2,50)
    
    E_v = []
    E_j = []
    sigma_array_v = []
    x_array_v = []
    check_v = []
    sigma_array_j = []
    x_array_j = []
    check_j = []
    
    j = 0
    for v in v_space:
        res = E.solve([v, j])
        E_v.append(res['gs_energy'])
        sigma = SigmaZ.expval(res['gs_vector'], check_real = True)
        x = XOperator.expval(res['gs_vector'], check_real = True)
        sigma_array_v.append(sigma)
        x_array_v.append(x)
        check_v.append(lam*sigma + x + j)
    
    v = 0
    for j in j_space:
        res = E.solve([v, j])
        E_j.append(res['gs_energy'])
        sigma = SigmaZ.expval(res['gs_vector'], check_real = True)
        x = XOperator.expval(res['gs_vector'], check_real = True)
        sigma_array_j.append(sigma)
        x_array_j.append(x)
        check_j.append(lam*sigma + x + j)
    
    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.suptitle('Groundstate energy in $v$ and $j$')
    
    axs[0].plot(v_space, E_v, label=r'$E$')
    axs[0].plot(v_space, sigma_array_v, label=r'$\sigma$')
    axs[0].plot(v_space, x_array_v, label=r'$\xi$')
    axs[0].plot(v_space, check_v, label=r'Eq. check')
    axs[0].legend()
    axs[0].set_xlabel(r'$v$')
    
    axs[1].plot(j_space, E_j, label=r'$E$')
    axs[1].plot(j_space, sigma_array_j, label=r'$\sigma$')
    axs[1].plot(j_space, x_array_j, label=r'$\xi$')
    axs[1].plot(j_space, check_j, label=r'Eq. check')
    axs[1].legend()
    axs[1].set_xlabel(r'$j$')
    
    plt.show()

def plot_functionals():
    sigma_space = np.linspace(-1,1,50)
    
    E_full = func.EnergyFunctional(H0_full, SigmaZ)
    E_KS = func.EnergyFunctional(H0_KS, SigmaZ)
    F_full = []
    F_full_eps = []
    F_KS = []
    E_c = []
    eps = 0.1
    
    for sigma in sigma_space:
        F_full.append(E_full.legendre_transform(sigma))
        F_full_eps.append(E_full.legendre_transform(sigma, epsMY=eps))
        F_KS.append(E_KS.legendre_transform(sigma))
        E_c.append(F_full[-1] - F_KS[-1])
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle('Universal functional in $\sigma$')
    ax.plot(sigma_space, F_full, 'b', label=r'$F^\lambda(\sigma,0)$, $\lambda={}$'.format(lam))
    ax.plot(sigma_space, F_full_eps, 'b--', label=r'$F_\varepsilon^\lambda(\sigma,0)$, $\lambda={}$, $\varepsilon={}$'.format(lam, eps))
    ax.plot(sigma_space, F_KS, 'g', label=r'$F^0(\sigma,0)$')
    ax.plot(sigma_space, 0.5-t*np.sqrt(1-np.square(sigma_space)), 'g.', label=r'$\frac{1}{2}-t\sqrt{1-\sigma^2}$')
    ax.plot(sigma_space, 0.5-t*np.sqrt(1-np.square(sigma_space)) - sigma_space*(lam**2)/2, 'b.', label=r'$\frac{1}{2}-t\sqrt{1-\sigma^2}-\frac{\sigma\lambda^2}{2}$')
    ax.plot(sigma_space, np.array(E_c), 'r', label=r'$E_c(\sigma,0)$')
    ax.legend()
    
    plt.show()

def plot_in_lambda(model: str): # model = Rabi | JC
    ## test lambda behavior
    lam_space = np.linspace(0,10,200)
    sigma_array = []
    x_array = []
    eig_array = []
    eig_diff = []
    for lam in lam_space:
        #print(f'lambda = {lam}')
        if model == 'Rabi':
            H0 = H0_KS + lam*CouplingRabi
        elif model == 'JC':
            H0 = H0_KS + lam*CouplingJC
        else:
            raise ValueError('No valid model specified.')
        
        GS = func.EnergyFunctional(H0).solve() # Hamiltonian, with coupling, no external coupling
        Psi0 = GS['gs_vector']
        sigma_array.append(SigmaZ.expval(Psi0, check_real = True))
        x_array.append(XOperator.expval(Psi0, check_real = True))
        eig_array.append(np.real(GS['eigval'][:10])) # lowest eigenvalues to see possible crossings
        eig_diff.append(GS['eigval'][1].real - GS['eigval'][0].real)
    
    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.suptitle(f'Expectation values in $\lambda$ ($t = {t}, j = {j}, v = {v}$)')
    axs[0].plot(lam_space, sigma_array)
    axs[0].set_xlabel(r'$\lambda$')
    axs[0].set_ylabel(r'$\sigma$')
    axs[1].plot(lam_space, x_array)
    axs[1].set_xlabel(r'$\lambda$')
    axs[1].set_ylabel(r'$x$')
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(f'Eigenvalues in $\lambda$ ($t = {t}, j = {j}, v = {v}$)')
    ax.plot(lam_space, eig_array)
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(f'Difference of two lowest eigenvalues in $\lambda$ ($t = {t}, j = {j}, v = {v}$)')
    ax.plot(lam_space, eig_diff)
    ax.set_yscale('log')
    
    plt.show()
    
