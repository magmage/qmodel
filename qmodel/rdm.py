'''
lib for RDM functions
Created on 11.06.2025
@author: mage
'''

import numpy as np
from .qmodel import Vector
from math import log

def rdm1(Psi: Vector):
    # return 1RDM as matrix in basis of Vector
    b1 = Psi.basis.part_basis # one-particle basis
    gamma = np.zeros((b1.dim, b1.dim), dtype=np.complex128)
    for index1,e1 in enumerate(b1.el):
        for index2,e2 in enumerate(b1.el):
            gamma[index1,index2] = Psi.basis.hop(e2,e1).expval(Psi)
    return gamma
           
def pair(Psi: Vector):
    # return diagonal of 2RDM (for interaction kernel) as matrix in basis of Vector
    b1 = Psi.basis.part_basis # one-particle basis
    p = np.zeros((b1.dim, b1.dim), dtype=np.complex128)
    for index1,e1 in enumerate(b1.el):
        for index2,e2 in enumerate(b1.el):
            p[index1,index2] = (Psi.basis.hop(e2,e2) - Psi.basis.hop(e2,e1) * Psi.basis.hop(e1,e2)).expval(Psi) # from CCR
    return p

def cum(Psi: Vector):
    gamma = rdm1(Psi)
    dens = np.diag(gamma)
    p = pair(Psi)
    return p + np.square(np.absolute(gamma)) - np.outer(dens, dens) # opposite sign as Ziesche1995

def entropy_fermions(Psi: Vector):
    non = np.linalg.eigh(rdm1(Psi)).eigenvalues # natural orbitals occupation numbers
    S = 0
    for n in non:
        try:
            S -= n*log(n) + (1-n)*log(1-n)
        except ValueError:
            pass
        except Exception as err:
            print(f"entropy_fermions(): Unexpected {err=}, {type(err)=}")
            raise
    return S