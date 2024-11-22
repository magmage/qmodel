# Hubbard trimer example reproducing some results (Fig. 1 and Fig. 2 (a)) from: (Spin-)density-functional theory for open-shell systems: Exact magnetization density functional for the half-filled Hubbard trimer, https://doi.org/10.1103/PhysRevA.100.012516

import matplotlib.pyplot as plt
import numpy as np
from qmodel import LatticeBasis, SpinBasis

M = 3 # lattice sites
N = 3 # particles

b_lattice = LatticeBasis(M)
b_spin = SpinBasis()
b_onepart = b_lattice.tensor(b_spin) # one-particle space
b = b_onepart.wedge(N)

# hopping and potential corresponding to Ullrich
t = 0.5
pot = [1,0,2]

# triangular Hamiltonian
next_site = lambda qn: dict(qn, i=qn['i']%M+1)
H_t = -t * b_onepart.sum(lambda qn: b.hop(next_site(qn), qn).add_adj()) + b_lattice.sum(lambda qni: pot[qni['i']-1] * b.hop(qni))

# triangular (t) and linear (l) energies and densities
E_t, E_l, n_t, n_l = [], [], [], []

# U-value range
U = np.linspace(0,10,1000)

# linear Hamiltonian created by substracting the end to end hopping
for u in U:
    H1_t = H_t + u * b_lattice.sum(lambda qni: b.hop(dict(qni, s=1)) * b.hop(dict(qni, s=-1)))
    H1_l = H1_t + t * b.hop({'i': 1}, {'i': M}).add_adj()
    res_t = H1_t.eig(hermitian = True)
    res_l = H1_l.eig(hermitian = True)
    E_t.append(res_t['eigenvalues'][0])
    E_l.append(res_l['eigenvalues'][0])
    n_t.append(res_t['eigenvectors'][0].trace(b_lattice))
    n_l.append(res_l['eigenvectors'][0].trace(b_lattice))

# plot the results
fig, axs = plt.subplots(2, 2, figsize = (10, 6))

axs[0,0].plot(U,E_t)
axs[0,1].plot(U,n_t)
axs[1,0].plot(U,E_l)
axs[1,1].plot(U,n_l)

for i in [0,1]:
    axs[i,0].set_ylabel('E')
    axs[i,1].set_ylabel('n')
    axs[0,i].set_title('triangle')
    axs[1,i].set_title('linear')
    for j in [0,1]:
        axs[i,j].set_xlabel('U')
        axs[i,j].set_xlim([np.min(U),np.max(U)])

plt.tight_layout()
plt.show()
