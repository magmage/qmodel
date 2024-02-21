import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


sys.path.append('../')
from qmodel import oscillator, spin, func
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.3)
#plt.rc('text', usetex=True)                                                     # Setting LaTex fonts and style
#plt.rc('font', family='serif', size=16)                                         # as well as the desired text size

colors = sns.color_palette('rocket', as_cmap=True)
sns.set(font_scale=1.5)

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
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11,7))
    fig.suptitle('Universal functional in $\sigma$')
    ax.plot(sigma_space, F_full, 'b', label=r'$F^\lambda(\sigma,0)$, $\lambda={}$'.format(lam))
    ax.plot(sigma_space, F_full_eps, 'b--', label=r'$F_\varepsilon^\lambda(\sigma,0)$, $\lambda={}$, $\varepsilon={}$'.format(lam, eps))
    ax.plot(sigma_space, F_KS, 'g', label=r'$F^0(\sigma,0)$')
    ax.plot(sigma_space, 0.5-t*np.sqrt(1-np.square(sigma_space)), 'g.', label=r'$\frac{1}{2}-t\sqrt{1-\sigma^2}$')
    ax.plot(sigma_space, 0.5-t*np.sqrt(1-np.square(sigma_space)) - sigma_space*(lam**2)/2, 'b.', label=r'$\frac{1}{2}-t\sqrt{1-\sigma^2}-\frac{\sigma\lambda^2}{2}$')
    ax.plot(sigma_space, np.array(E_c), 'r', label=r'$E_c(\sigma,0)$')
    ax.legend()
    fig.tight_layout()
    return fig, ax

fig,ax = plot_functionals()
plt.show()

