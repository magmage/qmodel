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

B = bf.diag()
T = A.tensor(B)
print(T)
print(T.basis)

## Distinguishable many-particle (qubit) systems

N = 2
b_single = SpinBasis()
b_many = b_single.ntensor(N) # full tensor product
print(b_many)

b_many_many = b_many.ntensor(2)
print(b_many_many)

Sz = b_single.sigma_z().extend(b_many)
print(Sz)
print(b_many.diag('s'))

b_spin1 = SpinBasis('s1')
b_spin2 = SpinBasis('s2')
b_many = b_spin1.tensor(b_spin2)
print(b_many)
print(b_spin1.sigma_z().extend(b_many))
print(b_many.diag('s1'))

## Fermionic many-particle systems

N = 2
b_single = SpinBasis()
b_many = b_single.wedge(N) # antisymmetric tensor product
print(b_many)
print(b_many.diag('s'))

b_single = Basis('x', (1,2,3)).tensor(SpinBasis())
print(b_single)
b_many = b_single.wedge(2)
print(b_many)

## Bosonic many-particle systems

N = 2
b_single = SpinBasis()
b_many = b_single.ntensor(N, 's') # symmetric tensor product
print(b_many)
print(b_many.diag('s'))

b_fermion = Basis('f', (1,2,3)).wedge(2)
b_boson = Basis('b', (1,2)).ntensor(2, 's')
print(b_fermion.tensor(b_boson))

b1_boson = NumberBasis(max_num=5, qn_key='n1')
print(b1_boson)
b2_boson = NumberBasis(max_num=5, qn_key='n2')
b_two_modes = b1_boson.tensor(b2_boson)
print(b_two_modes)

print(b1_boson.creator())
print(b1_boson.annihilator())
print(b1_boson.creator()*b1_boson.annihilator())
print(b1_boson.diag())
print(b1_boson.x_operator())
print(b1_boson.dx_operator())

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

## Dicke model example
import numpy as np
import matplotlib.pyplot as plt 

max_photon_num = 10
b_mode = NumberBasis(max_num=max_photon_num)
# define qubits separately so operators can act separately on them
b1_spin = SpinBasis('s1')
b2_spin = SpinBasis('s2')
b = b_mode.tensor(b1_spin).tensor(b2_spin)

num_op = b_mode.diag().extend(b) # number operator for photons
x_op = b_mode.x_operator().extend(b)
sigma_z1 = b1_spin.sigma_z().extend(b)
sigma_z2 = b2_spin.sigma_z().extend(b)
sigma_x1 = b1_spin.sigma_x().extend(b)
sigma_x2 = b2_spin.sigma_x().extend(b)

t = 1
H0 = num_op + 1/2 - t*(sigma_x1 + sigma_x2)
coupling_op = x_op*(sigma_z1 + sigma_z2)

spec = []
lam_span = np.linspace(0,5,50)
for lam in lam_span:
    H = H0 + lam*coupling_op
    res = H.eig(hermitian = True)
    spec.append(res['eigenvalues'][0:10])
plt.plot(lam_span, spec, 'b')
plt.show()