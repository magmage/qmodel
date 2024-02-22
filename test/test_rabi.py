'''
Created on 14.02.2024
@author: mage
'''
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from qmodel import oscillator, spin, func

# note these constants get changed in some procedures locally
t = 1
lam = 1
j = 0
v = 0
oscillator_size = 50

## Rabi model in 2nd quantization basis
OB = oscillator.Basis(oscillator_size)
SB = spin.Basis()
B = OB.tensor(SB)

XOperator = ((oscillator.Creator(OB) + oscillator.Annihilator(OB)) / sqrt(2)).extend(B)
dXOperator = ((-oscillator.Creator(OB) + oscillator.Annihilator(OB)) / sqrt(2)).extend(B)
SigmaX = spin.SigmaX(SB).extend(B)
SigmaZ = spin.SigmaZ(SB).extend(B)
ProjPlus = spin.ProjPlus(SB).extend(B)
ProjMinus = spin.ProjMinus(SB).extend(B)

CouplingRabi = SigmaZ*XOperator
CouplingJC = spin.SigmaMinus(SB).extend(B)*oscillator.Creator(OB).extend(B) + spin.SigmaPlus(SB).extend(B)*oscillator.Annihilator(OB).extend(B)

H0_Rabi_KS = -t*SigmaX + oscillator.Kinetic(OB).extend(B) # Rabi Hamiltonian, no coupling
H0_Rabi_full = H0_Rabi_KS + lam*CouplingRabi # Rabi Hamiltonian, with coupling
H0_JC_KS = t*SigmaZ + oscillator.Kinetic(OB).extend(B) # Jaynes-Cummings Hamiltonian, no coupling
H0_JC_full = H0_JC_KS + lam*CouplingJC # Jaynes-Cummings Hamiltonian, with coupling

def check_sigma_x(lam, sigma, x, j):
    # sigma and x must be in a precise relation after Eq. (33) of draft
    if abs(lam*sigma + j + x) > 10e-4:
        print(f'sigma--xi check: FAIL at lam={lam}, sigma={sigma}, xi={x}, j={j}! maybe increase oscillator_size value')
        

def plot_photon_filling():
    Psi0 = func.EnergyFunctional(H0_Rabi_full, [SigmaZ, XOperator]).solve([v, j])['gs_vector']
    
    sigma_expval = SigmaZ.expval(Psi0, check_real = True)
    x_expval = XOperator.expval(Psi0, check_real = True)
    print('sigma_z expectation value = {}'.format(sigma_expval))
    print('x expectation value = {}'.format(x_expval))
    check_sigma_x(lam, sigma_expval, x_expval, j)
    
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
    #E = func.EnergyFunctional(H0_Rabi_full, [SigmaZ, XOperator])
    E1 = func.EnergyFunctional(H0_Rabi_full, SigmaZ)
    E2 = func.EnergyFunctional(H0_Rabi_full, XOperator)

    v_space = np.linspace(-2,2,50)
    j_space = np.linspace(-2,2,50)
    
    E_v = []
    E_j = []
    sigma_array_v = []
    x_array_v = []
    check_v = []
    check_v_diff = []
    sigma_array_j = []
    x_array_j = []
    check_j = []
    
    j = 0
    for v in v_space:
        res = E1.solve(v) # E1.solve([v, j])
        E_v.append(res['gs_energy'])
        sigma = SigmaZ.expval(res['gs_vector'], check_real = True)
        x = XOperator.expval(res['gs_vector'], check_real = True)
        sigma_array_v.append(sigma)
        x_array_v.append(x)
        check_v.append(lam*sigma + x + j)
        # test legendre trafo
        check_v_diff.append(v - E1.legendre_transform(sigma)['pot'])
    
    v = 0
    for j in j_space:
        res = E2.solve(j)
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
    axs[0].plot(v_space, check_v, '--', label=r'Eq. check')
    axs[0].plot(v_space, check_v_diff, '--', label=r'Legendre check')
    axs[0].legend()
    axs[0].set_xlabel(r'$v$')
    
    axs[1].plot(j_space, E_j, label=r'$E$')
    axs[1].plot(j_space, sigma_array_j, label=r'$\sigma$')
    axs[1].plot(j_space, x_array_j, label=r'$\xi$')
    axs[1].plot(j_space, check_j, '--', label=r'Eq. check')
    axs[1].legend()
    axs[1].set_xlabel(r'$j$')
    
    plt.show()

def plot_functionals_in_sigma():
    sigma_space = np.linspace(-1,1,51)
    
    lam = 1
    x = 0.5
    eps = 0.1
    
    E_full = func.EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [SigmaZ, XOperator])
    E_KS = func.EnergyFunctional(H0_Rabi_KS, [SigmaZ, XOperator])
    F_full = []
    F_full_eps = []
    F_KS = []
    E_xc = []
    test_F_full = []
    test_F_KS = []
    W_0 = []
    W_c = []
    T_c = []
    test_E_xc = []
    
    for sigma in sigma_space:
        LT_full = E_full.legendre_transform([sigma, x])
        LT_full_eps = E_full.legendre_transform([sigma, x], epsMY=eps)
        LT_KS = E_KS.legendre_transform([sigma, x])
        
        check_sigma_x(lam, sigma, x, LT_full['pot'][1])
        check_sigma_x(0, sigma, x, LT_KS['pot'][1])
        
        F_full.append(LT_full['F'])
        F_full_eps.append(LT_full_eps['F'])
        F_KS.append(LT_KS['F'])
        E_xc.append(F_full[-1] - F_KS[-1])
        
        # solve KS and full system
        sol_full = E_full.solve(LT_full['pot'])
        sol_KS = E_KS.solve(LT_KS['pot'])
        
        test_F_full.append(sol_full['gs_energy'] - np.dot(LT_full['pot'], [sigma, x]))
        test_F_KS.append(sol_KS['gs_energy'] - np.dot(LT_KS['pot'], [sigma, x]))
        # get correlation parts
        W_0.append(lam*CouplingRabi.expval(sol_KS['gs_vector'], check_real=True))
        W_c.append(lam*CouplingRabi.expval(sol_full['gs_vector'], check_real=True) - W_0[-1])
        T_c.append(H0_Rabi_KS.expval(sol_full['gs_vector'], check_real=True) - H0_Rabi_KS.expval(sol_KS['gs_vector'], check_real=True))
        test_E_xc.append(W_0[-1] + (W_c[-1] + T_c[-1]))
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(r'Universal functional in $\sigma$ at $\xi$={}'.format(x))
    ax.plot(sigma_space, F_full, 'b', label=r'$F^\lambda(\sigma,\xi)$, $\lambda={}$'.format(lam))
    ax.plot(sigma_space, F_full_eps, 'b--', label=r'$F_\varepsilon^\lambda(\sigma,\xi)$, $\lambda={}$, $\varepsilon={}$'.format(lam, eps))
    ax.plot(sigma_space, F_KS, 'g', label=r'$F^0(\sigma,\xi)$')
    ax.plot(sigma_space, 0.5-t*np.sqrt(1-np.square(sigma_space))+x**2/2, 'g.', label=r'$\frac{1}{2}-t\sqrt{1-\sigma^2}+\frac{1}{2}\xi^2$')
    #approx = lambda sigma: 0.5-t*sqrt(1-sigma**2)+lam*x*sigma+x**2/2-abs(sigma)*(lam**2)/2
    #ax.plot(sigma_space, list(map(approx, sigma_space)), 'b', label=r'$\frac{1}{2}-t\sqrt{1-\sigma^2}+\lambda\sigma\xi+\frac{1}{2}\xi^2-\frac{|\sigma|\lambda^2}{2}$')
    #ax.plot(sigma_space, func.moreau_envelope(sigma_space, approx, eps), 'b--', label=r'MY')
    ax.plot(sigma_space, E_xc, 'r', label=r'$E_{xc}(\sigma,\xi)$')
    ax.plot(sigma_space, test_F_full, 'b.')
    ax.plot(sigma_space, test_F_KS, 'g.')
    ax.plot(sigma_space, W_0, 'k--', label=r'$\lambda W^0(\sigma,\xi)$')
    ax.plot(sigma_space, W_c, 'c--', label=r'$\lambda W_c(\sigma,\xi)$')
    ax.plot(sigma_space, T_c, 'c', label=r'$T_c(\sigma,\xi)$')
    ax.plot(sigma_space, test_E_xc, 'r.')
    ax.legend()
    
    plt.show()

def plot_functionals_in_lambda():
    sigma = 0.3
    x = 0.5
    lam_space = np.linspace(0,2,30)

    # solve KS system
    E_KS = func.EnergyFunctional(H0_Rabi_KS, [SigmaZ, XOperator])
    LT_KS = E_KS.legendre_transform([sigma, x])
    check_sigma_x(0, sigma, x, LT_KS['pot'][1])
    sol_KS = E_KS.solve(LT_KS['pot'])
    CouplingKS = CouplingRabi.expval(sol_KS['gs_vector'], check_real=True)
    
    F_full = []
    F_full_eps = []
    F_KS = []
    E_xc = []
    test_F_full = []
    W_0 = []
    W_c = []
    T_c = []
    test_E_xc = []
    eps = 0.1
    
    for lam in lam_space:
        E_full = func.EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [SigmaZ, XOperator])
        LT_full = E_full.legendre_transform([sigma, x])
        check_sigma_x(lam, sigma, x, LT_full['pot'][1])
        LT_full_eps = E_full.legendre_transform([sigma, x], epsMY=eps)
        
        F_full.append(LT_full['F'])
        F_full_eps.append(LT_full_eps['F'])
        F_KS.append(LT_KS['F'])
        E_xc.append(F_full[-1] - F_KS[-1])
        # solve full system
        sol_full = E_full.solve(LT_full['pot'])
        test_F_full.append(sol_full['gs_energy'] - np.dot(LT_full['pot'], [sigma, x]))
        # get correlation parts
        W_0.append(lam*CouplingKS)
        W_c.append(lam*CouplingRabi.expval(sol_full['gs_vector'], check_real=True) - W_0[-1])
        T_c.append(H0_Rabi_KS.expval(sol_full['gs_vector'], check_real=True) - H0_Rabi_KS.expval(sol_KS['gs_vector'], check_real=True))
        test_E_xc.append(W_0[-1] + (W_c[-1] + T_c[-1]))
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(r'Universal functional in $\lambda$ at $\sigma={}$ and $\xi={}$'.format(sigma, x))
    ax.plot(lam_space, F_full, 'b', label=r'$F^\lambda(\sigma,\xi)$')
    ax.plot(lam_space, F_full_eps, 'b--', label=r'$F_\varepsilon^\lambda(\sigma,\xi)$, $\varepsilon={}$'.format(eps))
    ax.plot(lam_space, F_KS, 'g', label=r'$F^0(\sigma,\xi)$')
    ax.plot(lam_space, E_xc, 'r', label=r'$E^\lambda_{xc}(\sigma,\xi)$')
    ax.plot(lam_space, test_F_full, 'b.')
    ax.plot(lam_space, W_0, 'k--', label=r'$\lambda W^0(\sigma,\xi)$')
    ax.plot(lam_space, W_c, 'c--', label=r'$\lambda W^\lambda_c(\sigma,\xi)$')
    ax.plot(lam_space, T_c, 'c', label=r'$T^\lambda_c(\sigma,\xi)$')
    ax.plot(lam_space, test_E_xc, 'r.')
    ax.legend()
    
    plt.show()
    
def plot_in_lambda(model: str): # model = Rabi | JC
    ## test lambda behavior
    lam_space = np.linspace(0,7,300)
    sigma_array = []
    x_array = []
    eig_array = []
    eig_diff = []
    for lam in lam_space:
        #print(f'lambda = {lam}')
        if model == 'Rabi':
            H0 = H0_Rabi_KS + lam*CouplingRabi
        elif model == 'JC':
            H0 = H0_JC_KS + lam*CouplingJC
        else:
            raise ValueError('No valid model specified.')
        
        GS = func.EnergyFunctional(H0).solve() # Hamiltonian, with coupling, no external potential
        Psi0 = GS['gs_vector']
        sigma_array.append(SigmaZ.expval(Psi0, check_real = True))
        x_array.append(XOperator.expval(Psi0, check_real = True))
        eig_array.append(np.real(GS['eigval'][:10])) # lowest eigenvalues to see possible crossings
        eig_diff.append(GS['eigval'][1].real - GS['eigval'][0].real)
    
    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.suptitle(f'{model}: Expectation values in $\lambda$ ($t = {t}, j = {j}, v = {v}$)')
    axs[0].plot(lam_space, sigma_array)
    axs[0].set_xlabel(r'$\lambda$')
    axs[0].set_ylabel(r'$\sigma$')
    axs[1].plot(lam_space, x_array)
    axs[1].set_xlabel(r'$\lambda$')
    axs[1].set_ylabel(r'$x$')
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(f'{model}: Eigenvalues in $\lambda$ ($t = {t}, j = {j}, v = {v}$)')
    ax.plot(lam_space, eig_array)
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(f'{model}: Difference of two lowest eigenvalues in $\lambda$ ($t = {t}, j = {j}, v = {v}$)')
    ax.plot(lam_space, eig_diff)
    ax.set_yscale('log')
    
    plt.show()
    
