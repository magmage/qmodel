'''
extension library for DFT objects
(CC0) Markus Penz
'''

import numpy as np
from typing import Union
from scipy.optimize import minimize
from .qmodel import Operator, OperatorList

class EnergyFunctional:
    """
    A class representing the energy functional with a base Hamiltonian H0 and operators
    that give 'density' variables as expectation values.

    This class provides methods for solving the Hamiltonian, performing Legendre transforms,
    and proximal mappings for energy functionals.
    
    Attributes:
        H0 (Operator): The base Hamiltonian.
        dens_operators (Union[Operator, OperatorList]): Operators related to density variables.
    """
    """
    A class representing the energy functional with a base Hamiltonian H0 and operators
    that give 'density' variables as expectation values.

    This class provides methods for solving the Hamiltonian, performing Legendre transforms,
    and proximal mappings for energy functionals.
    
    Attributes:
        H0 (Operator): The base Hamiltonian.
        dens_operators (Union[Operator, OperatorList]): Operators related to density variables.
    """
    
    def __init__(self, H0: Operator, dens_operators: Union[Operator, OperatorList, list]):
        """
        Initializes the energy functional with a base Hamiltonian and density operators.

        Parameters:
            H0 (Operator): The base Hamiltonian.
            dens_operators (Union[Operator, OperatorList, list]): The density operators used in the functional.

        Raises:
            TypeError: If H0 is not of type 'Operator'.
            ValueError: If the bases of H0 and dens_operators do not match.
        """
        if not isinstance(H0, Operator):
            raise TypeError('Base Hamiltonian must be of type Operator')
        if isinstance(dens_operators, list):
            dens_operators = OperatorList(H0.basis, dens_operators) # convert to OperatorList using H0 basis
        if dens_operators is not None and H0.basis != dens_operators.basis:
            raise ValueError('Base Hamiltonian and density operators must share same basis.')
        
        self.H0 = H0
        self.dens_operators = dens_operators
    
    def solve(self, pot: Union[float, list]):
        """
        Solves the Hamiltonian for a given potential and returns the ground state energy, vector, and degeneracy.

        Parameters:
            pot (Union[float, list]): The potential value or a list of values.
        
        Returns:
            dict: A dictionary containing the ground state energy, vector, and degeneracy.
        """
        H = self.H0 + self.dens_operators*pot
        sol = H.eig(hermitian = True)
        return {**sol, **{
                'gs_energy': sol['eigenvalues'][0],
                'gs_vector': sol['eigenvectors'][0],
                'gs_degeneracy': sol['degeneracy'][0]
            }} # merge with solution dict
    
    #@timer
    def legendre_transform(self, dens: Union[float, list], epsMY: float = 0, verbose = False):
        """
        Computes the Legendre transform of the energy functional at a given density.

        Parameters:
            dens (Union[float, list]): The density at which the Legendre transform is computed.
            epsMY (float): A regularization term.
            verbose (bool): If True, prints a warning if optimization fails.

        Returns:
            dict: A dictionary containing the Legendre transform (F) and the corresponding potential (pot).

        Raises:
            ValueError: If the number of density components and density operators do not match.
        """    
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
        """
        Performs the proximal mapping on the energy functional, using the Legendre transform.

        Parameters:
            dens (Union[float, list]): The density value.
            epsMY (float): The regularization parameter.

        Returns:
            float: The result of the proximal mapping.

        Raises:
            ValueError: If the number of density components and density operators do not match.
        """        
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
    """
    Maps a function over an iterable and returns the result as a NumPy array.

    Parameters:
        func (Callable): The function to apply to each element.
        xiter (Iterable): The iterable over which to apply the function.

    Returns:
        np.ndarray: The result of applying the function to each element.
    """
    return np.array(list(map(func, xiter))) # map a function on iterator and return nparray
    
def moreau_envelope(xspan, func, epsMY: float):
    """
    Computes the Moreau envelope on a 1D NumPy array with the given function and regularization parameter.

    Parameters:
        xspan (np.ndarray): The input array over which the envelope is computed.
        func (Callable): The function to apply to each element.
        epsMY (float): The regularization parameter.

    Returns:
        np.ndarray: The result of the Moreau envelope computation.
    """
    # on 1d nparray xspan with function evaluated by func(x) at each x
    # xspan has to be symmetric around 0
    return np_map( lambda x : min(np_map(func, xspan) + (xspan-x)**2/(2*epsMY)), xspan )
