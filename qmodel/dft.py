## library for DFT objects
## written for Oslo Dicke-model project
## adapted from qmodel_old/func.py

import numpy as np
from typing import Union
from scipy.optimize import minimize
from .qmodel import Operator, OperatorList, Vector
from .timer import timer

class EnergyFunctional:
    # energy functional with base Hamiltonian H0
    # and operators that give 'density' variable as expectation values
    
    def __init__(self, H0: Operator, dens_operators: Union[Operator, OperatorList, list]):
        if not isinstance(H0, Operator):
            raise TypeError('Base Hamiltonian must be of type Operator')
        if isinstance(dens_operators, list):
            dens_operators = OperatorList(H0.basis, dens_operators) # convert to OperatorList using H0 basis
        if dens_operators is not None and H0.basis != dens_operators.basis:
            raise ValueError('Base Hamiltonian and density operators must share same basis.')
        
        self.H0 = H0
        self.dens_operators = dens_operators
    
    def solve(self, pot: Union[float, list]):
        H = self.H0 + self.dens_operators*pot
        sol = H.eig(hermitian = True)
        return {**sol, **{
                'gs_energy': sol['eigenvalues'][0],
                'gs_vector': sol['eigenvectors'][0],
                'gs_degeneracy': sol['degeneracy'][0]
            }} # merge with solution dict
    
    #@timer
    def legendre_transform(self, dens: Union[float, list], epsMY: float = 0, verbose = False):
        # gives Legendre transform F at dens, also returns potential (density-potential inversion)
        # dens must match dens_operators
        if (isinstance(dens, float) and len(self.dens_operators) > 1) or (isinstance(dens, list) and len(self.dens_operators) == 1):
            raise ValueError('Numbers of density components and density operators in energy functional must agree.')
        # F from E functional
        def G(pot):
            if len(self.dens_operators) == 1: pot = pot[0] # just single component
            # already include MY regularization in E functional after Eq. (10) in doi:10.1063/1.5037790 (Generalized KS on BS)
            return -(self.solve(pot)['gs_energy'] - epsMY*np.linalg.norm(pot)**2/2 - np.dot(pot, dens))
        # perform optimization
        res = minimize(G, [0]*len(self.dens_operators), method='BFGS', options={'disp': False, 'gtol': 1e-5})
        if res['success'] == False and verbose: print('Warning! Optimization in Legendre transformation not successful.')
        pot = res['x'] if len(res['x']) > 1 else res['x'][0]
        return {'F': -res['fun'], 'pot': pot}
    
    #@timer
    def prox(self, dens: Union[float, list], epsMY: float):
        # slow since it must perform legendre_transform *inside* optimization!
        # dens must match dens_operators
        if (isinstance(dens, float) and len(self.dens_operators) > 1) or (isinstance(dens, list) and len(self.dens_operators) == 1):
            raise ValueError('Numbers of density components and density operators in energy functional must agree.')
        # proximal mapping
        def G(dens2):
            return epsMY*self.legendre_transform(dens2)['F'] + np.linalg.norm(dens - dens2)**2/2
        # perform optimization
        # start with 0 (fails with dens)
        res = minimize(G, [0]*len(self.dens_operators), method='BFGS', options={'disp': False, 'gtol': 1e-5})
        if res['success'] == False: print('Warning! Optimization in proximal mapping not successful.')
        return res['x'] if len(res['x']) > 1 else res['x'][0]

def np_map(func, xiter):
    return np.array(list(map(func, xiter))) # map a function on iterator and return nparray
    
def moreau_envelope(xspan, func, epsMY: float):
    # on 1d nparray xspan with function evaluated by func(x) at each x
    # xspan has to be symmetric around 0
    return np_map( lambda x : min(np_map(func, xspan) + (xspan-x)**2/(2*epsMY)), xspan )
