## library for DFT objects
## written for Oslo QEDFT project / 2024, Feb 15

import numpy as np
from typing import Union
from scipy.optimize import minimize
from .hilb import Operator, OperatorList, Vector
from .timer import tictoc

class EnergyFunctional:
    # energy functional with base Hamiltonian H0
    # and operators that give 'density' variable as expectation values
    
    def __init__(self, H0: Operator, dens_operators: Union[Operator, OperatorList, list] = None):
        if not isinstance(H0, Operator):
            raise TypeError('Base Hamiltonian must be of type Operator')
        if isinstance(dens_operators, list):
            dens_operators = OperatorList(H0.basis, dens_operators) # convert to OperatorList 
        if dens_operators is not None and H0.basis != dens_operators.basis:
            raise ValueError('Base Hamiltonian and density operators must share same basis.')
        
        self.H0 = H0
        self.dens_operators = dens_operators
    
    def solve(self, pot: Union[float, list] = 0, verbose = True):
        # get eigenvalues, eigenvectors, groundstate energy etc. by exact diagonalization
        # first assemble Hamiltonian
        H = self.H0.copy()
        if self.dens_operators is not None:
            H = H + self.dens_operators*pot
        eigval,eigvec = np.linalg.eigh(H.matrix)
        eigvec = np.transpose(eigvec) # first index is eigvec number
        gs_energy = eigval[0]
        gs_vector = Vector(H.basis, eigvec[0,:] / np.linalg.norm(eigvec[0,:])) # normalize

        # check for ground-state degeneracy (within num accuracy)
        gs_degen = np.count_nonzero(np.abs(eigval - gs_energy) < 1e-6)
        if verbose and gs_degen > 1:
            print(f'Groundstate has {gs_degen}-fold degeneracy. Ground-state vector is thus non-unique.')
        
        return {
                'gs_energy': gs_energy,
                'gs_vector': gs_vector,
                'gs_degen': gs_degen,
                'eigval': eigval,
                'eigvec': eigvec
            }
    
    @tictoc
    def legendre_transform(self, dens: Union[float, list], epsMY: float = 0):
        # dens must match dens_operators
        if (isinstance(dens, float) and len(self.dens_operators) > 1) or (isinstance(dens, list) and len(self.dens_operators) == 1):
            raise ValueError('Numbers of density components and density operators in energy functional must agree.')
        # F from E functional
        def G(v):
            if len(self.dens_operators) == 1: v = v[0] # just single component
            # already include MY regularization in E functional after Eq. (10) in doi:10.1063/1.5037790
            return -(self.solve(v)['gs_energy'] - epsMY*np.linalg.norm(v)**2/2 - np.dot(dens, v))
        # perform optimization
        res = minimize(G, [0]*len(self.dens_operators), method='BFGS', options={'disp': False, 'gtol': 1e-5})
        if res['success'] == False: print('Warning! Optimization in Legendre transformation not successful.')
        return -res['fun']
    
    @tictoc
    def prox(self, dens: Union[float, list], epsMY: float):
        ## must still be tested!
        # dens must match dens_operators
        if (isinstance(dens, float) and len(self.dens_operators) > 1) or (isinstance(dens, list) and len(self.dens_operators) == 1):
            raise ValueError('Numbers of density components and density operators in energy functional must agree.')
        # proximal mapping
        def G(dens2):
            return epsMY*self.legendre_transform(dens2) + np.linalg.norm(dens - dens2)**2/2
        # perform optimization
        res = minimize(G, dens, method='BFGS', options={'disp': False, 'gtol': 1e-5})
        if res['success'] == False: print('Warning! Optimization in proximal mapping not successful.')
        return res['x'] if len(res['x']) > 1 else res['x'][0]