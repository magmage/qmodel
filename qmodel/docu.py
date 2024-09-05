'''
Created on 12.08.2024

@author: mage

code for documentation
'''
from qmodel import *

## Getting started

b = Basis('i', (1,2,3))
print(b)

A = b.diag()
print(A)

B = 3*A - A/2 + 1
print(A) # stays the same
print(B)

## Tensor products

bi = Basis("i", (1,2,3))
bf = Basis("f", (-1,0,1))
b1 = bi.tensor(bf)
b2 = bf.tensor(bi)
print(b1)
print(b2)

A = bi.diag()
print(A.extend(b1))
print(A.extend(b2))

## Spin-lattice example

M = 3
b_lattice = LatticeBasis(M)
b_spin = SpinBasis()
b_onepart = b_lattice.tensor(b_spin) # one-particle space
print(b_onepart)

N = 2
b = b_onepart.wedge(N)
print(b)

b_lattice.sum(lambda qn: print(qn))
b_onepart.sum(lambda qn: print(qn))

t = 1
U = 1
next_site = lambda qn: dict(qn, i=qn['i']%M+1)
H = -t * b_onepart.sum(lambda qn: b.hop(next_site(qn), qn).add_adj()) \
    +U * b_lattice.sum(lambda qni: b.hop(dict(qni, s=1)) * b.hop(dict(qni, s=-1))) 
print(H)

E = H.eig(hermitian = True)
gs_energy = E['eigenvalues'][0]
gs_vector = E['eigenvectors'][0]
gs_degeneracy = E['degeneracy'][0]
gs_density = gs_vector.trace(b_lattice)
print(gs_energy)
print(gs_density)
print(gs_degeneracy)

pot = [0, -1, 0]
H = H + b_lattice.sum(lambda qni: pot[qni['i']-1] * b.hop(qni))
E = H.eig(hermitian = True)
gs_energy = E['eigenvalues'][0]
gs_vector = E['eigenvectors'][0]
gs_density = gs_vector.trace(b_lattice)
print(gs_energy)
print(gs_density)

S = b_spin.sigma_vec().extend(b)
print(S.expval(gs_vector))