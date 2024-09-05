'''
Created on 10.08.2024

@author: mage

spin-lattice example from documentation / changed
'''
from qmodel import *
import matplotlib.pyplot as plt
import numpy as np

# compare F for dimer to https://arxiv.org/pdf/2303.15084

M = 3 # lattice sites
N = 3 # particles
b_lattice = LatticeBasis(M)
b_spin = SpinBasis()
print(b_spin)
b_onepart = b_lattice.tensor(b_spin) # one-particle space
b = b_onepart.wedge(N)

t = 1
U = 1
next_site = lambda qn: dict(qn, i=qn['i']%M+1)

H = -t * b_onepart.sum(lambda qn: b.hop(next_site(qn), qn).add_adj()) \
    +U * b_lattice.sum(lambda qni: b.hop(dict(qni, s=1/2)) * b.hop(dict(qni, s=-1/2))) 

E = H.eig(hermitian = True)

gs_energy = E['eigenvalues'][0]
gs_vector = E['eigenvectors'][0]
gs_density = gs_vector.trace(b_lattice)
print(gs_energy)
print(gs_density)
print(E['degeneracy'])

pot = [0, -1, 0]
H = H + b_lattice.sum(lambda qni: pot[qni['i']-1] * b.hop(qni))
E = H.eig(hermitian = True)
gs_energy = E['eigenvalues'][0]
gs_vector = E['eigenvectors'][0]
gs_density = gs_vector.trace(b_lattice)
print(gs_energy)
print(gs_density)
print(E['degeneracy'])

Sx = b_spin.sigma_x().extend(b) # acts on all lattice sites
Sx_exp_arr = []
lam_arr = np.linspace(-5,5,41)
for lam in lam_arr:
    H1 = H + lam*Sx
    E = H1.eig(hermitian = True)
    gs_vector = E['eigenvectors'][0]
    Sx_exp_arr.append(Sx.expval(gs_vector, transform_real=True))
plt.plot(lam_arr, Sx_exp_arr)
plt.show()