'''
Created on 11.08.2024

@author: mage

test suite for qmodel library
some tests use results from the one before
'''
from qmodel import *

eps = 1e-15 # accuracy for tests

def numdelta(n1, n2):
    d = abs(n1-n2)
    if d > eps: print('failed!')
    else: print('passed')

def opdelta(op1, op2):
    d = (op1 - op2).norm()
    if d > eps: print('failed!')
    else: print('passed')

b1 = Basis('key1', range(3))
b2 = Basis('key2', range(3))
b = b1.tensor(b2)
op1 = b.diag('key1')
op2 = b1.diag().extend(b)
print('Test 1:')
opdelta(op1, op2)

b = b1.ntensor(2)
op1 = b1.diag().extend(b)
op2 = b1.diag().tensor(b1.id(), b) + b1.id().tensor(b1.diag(), b)
print('Test 2:')
opdelta(op1, op2)

op3 = b.diag() # gives same operator
print('Test 3:')
opdelta(op1, op3)

# 5-site lattice Hamiltonian
# diag vs. hop
M = 5
b_lattice = LatticeBasis(M)
op1 = b_lattice.diag()
op2 = b_lattice.sum(lambda qn: qn['i']*b_lattice.hop(qn))
print('Test 4:')
opdelta(op1, op2)

next_site = lambda qn: dict(qn, i=qn['i']%M+1)
H = -b_lattice.sum(lambda qn: b_lattice.hop(next_site(qn), qn).add_adj()) + op1
E = H.eig(hermitian = True)
gs_energy = E['eigenvalues'][0]
gs_vector = E['eigenvectors'][0]
print('Test 5:')
numdelta(gs_energy, H.expval(gs_vector))

## successfully tested tretrahedron graph anderson impurity against ex-tetrahedron-Anderson.py