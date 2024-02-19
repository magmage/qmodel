'''
Created on 18.02.2024
@author: mage
'''
import numpy as np
from .hilb import Basis as HilbBasis
from .hilb import Operator as HilbOperator


class Basis(HilbBasis):
    # default oscillator basis
    
    def __init__(self, size = 2, qn_key = 'number_n'):
        super().__init__( (qn_key, range(0, size)) )

        
class Operator(HilbOperator):
    # default oscillator operator
    
    def __init__(self, basis: Basis):
        if not isinstance(basis, Basis):
            raise TypeError('The basis assigned to the operator is not an oscillator basis.')
        
        super().__init__(basis)
    

class Hamiltonian(Operator):
    # pure oscillator Hamiltonian
    
    def __init__(self, basis: Basis, om = 1):
        self.om = om
        super().__init__(basis)

    def create(self):  
        # generate matrix / only diagonal
        self.matrix = self.om * (np.diag(np.array(range(0,self.basis.dim)) + 1/2))

      
class Creator(Operator):
    def create(self):
        # generate matrix / only -1 secondary-diagonal in qn
        if self.basis.dim > 1:
            self.matrix = np.diag(np.sqrt(range(1,self.basis.dim)), -1)
        else:
            self.matrix = np.identity(1, dtype=np.complex_)
            

class Annihilator(Operator):
    def create(self):
        # generate matrix / only +1 secondary-diagonal in qn
        self.matrix = np.diag(np.sqrt(range(1,self.basis.dim)), +1)