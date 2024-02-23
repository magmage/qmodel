import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


sys.path.append('../')
from qmodel import oscillator, spin, func
import seaborn as sns

plt.style.use('seaborn-v0_8')
plt.rc('text', usetex=True)                                                     # Setting LaTex fonts and style
plt.rc('font', family='serif', size=40)                                         # as well as the desired text size

sns.set(font_scale=2)
plt.rcParams['figure.facecolor'] = 'none'
plt.rc('lines', linewidth=2,markersize=10)

t = 1
lam = 1
j = 0
v = 0.02
oscillator_size = 10

## Rabi model in 2nd quantization basis
OB = oscillator.Basis(oscillator_size)
SB = spin.Basis()
B = OB.tensor(SB)

XOperator = ((oscillator.Creator(OB) + oscillator.Annihilator(OB)) / sqrt(2)).extend(B)
SigmaX = spin.SigmaX(SB).extend(B)
SigmaZ = spin.SigmaZ(SB).extend(B)

CouplingRabi = SigmaZ*XOperator
H0_KS = -t*SigmaX + oscillator.Kinetic(OB).extend(B) # Hamiltonian, just kinetic
H0_full = H0_KS + lam*CouplingRabi # Hamiltonian, with coupling
H = H0_full + j*XOperator + v*SigmaZ # Hamiltonian, with external potential and current

def plot_functionals():
    sigma_space = np.linspace(-1,1,201)
#    
#    E_full = func.EnergyFunctional(H0_full, SigmaZ)
#    E_KS = func.EnergyFunctional(H0_KS, SigmaZ)
#    F_full = []
#    F_full_eps = []
#    F_KS = []
#    E_c = []
#    eps = 0.1
#    
#    for sigma in sigma_space:
#        F_full.append(E_full.legendre_transform(sigma))
#        F_full_eps.append(E_full.legendre_transform(sigma, epsMY=eps))
#        F_KS.append(E_KS.legendre_transform(sigma))
#        E_c.append(F_full[-1] - F_KS[-1])
    lam = 1
    x = 0
    eps = 0.1

    E_full = func.EnergyFunctional(H0_KS + lam*CouplingRabi, [SigmaZ, XOperator])
    E_KS = func.EnergyFunctional(H0_KS, [SigmaZ, XOperator])
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
        
        #check_sigma_x(lam, sigma, x, LT_full['pot'][1])
        #check_sigma_x(0, sigma, x, LT_KS['pot'][1])
        
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
        T_c.append(H0_KS.expval(sol_full['gs_vector'], check_real=True) - H0_KS.expval(sol_KS['gs_vector'], check_real=True))
        test_E_xc.append(W_0[-1] + (W_c[-1] + T_c[-1])) 

    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(16,9))
    ax1.set_title(r'Universal functional - Non-interacting system ($\lambda=0$)',fontsize=40)
    ax1.set_xlabel(r'$\sigma$')
    ax1.set_ylabel(r'$\mathcal{F}_{LL}(\sigma,0)$')

    fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(16,9))
    ax2.set_title(r'Universal functional - Interacting system ($\lambda=1$)',fontsize=40)
    ax2.set_xlabel(r'$\sigma$')
    ax2.set_ylabel(r'$\mathcal{F}_{LL}(\sigma,0)$')

    # Plot KS
    ax1.plot(sigma_space, 0.5-t*np.sqrt(1-np.square(sigma_space)), label=r'Analytic Approx')
    ax1.plot(sigma_space, F_KS, '--', label=r'Numerical')
    
    # Plot fully interacting
    ax2.plot(sigma_space, 0.5-t*np.sqrt(1-np.square(sigma_space)) - (1-sigma_space**2)*(lam**2)/4, label=r'Analytic Approx')
    ax2.plot(sigma_space, F_full,'.', label=r'Numerical')
    ax2.plot(sigma_space, F_full_eps, '.', label=r'$F_\varepsilon^{\lambda=1}(\sigma,0)$' + fr', $\varepsilon={eps}$')
    #ax.plot(sigma_space, np.array(E_c), 'r', label=r'$E_c(\sigma,0)$')
    ax2.plot(sigma_space, 0.5 - t*np.sqrt(1-sigma_space**2) - 1*lam**2/4, label=r'Lower bound')

    ax1.legend()
    ax2.legend()

    fig1.tight_layout()
    fig2.tight_layout()

    return [fig1,fig2], [ax1,ax2]

fig, ax = plot_functionals()

fig[0].savefig('Plots/UniversalFunctionalNonInteracting.pdf')
fig[1].savefig('Plots/UniversalFunctionalInteracting.pdf')
plt.show()

