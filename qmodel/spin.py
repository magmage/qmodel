'''
Created on 18.02.2024
@author: mage
'''

from .hilb import Basis as HilbBasis
from .hilb import Operator as HilbOperator
from .hilb import OperatorList


class Basis(HilbBasis):
    # default spin 1/2 basis
    
    def __init__(self, qn_key = 'spin_s'):
        super().__init__( (qn_key, (+1/2,-1/2)) )

        
class Operator(HilbOperator):
    # default spin operator
    
    def __init__(self, basis: Basis):
        if not isinstance(basis, Basis):
            raise TypeError('The basis assigned to the operator is not a spin basis.')
        
        super().__init__(basis)
    

class SigmaX(Operator):
    def create(self):
        self.matrix[0, 1] = 1
        self.matrix[1, 0] = 1


class SigmaY(Operator):
    def create(self):
        self.matrix[0, 1] = -1j
        self.matrix[1, 0] = 1j

        
class SigmaZ(Operator):
    def create(self):
        self.matrix[0, 0] = 1
        self.matrix[1, 1] = -1

        
class SigmaVec(OperatorList):
    def __init__(self, basis: Basis):
        super().__init__(basis, [SigmaX(basis), SigmaY(basis), SigmaZ(basis)])


class SigmaPlus(Operator):
    def create(self):
        self.matrix[0, 1] = 1


class SigmaMinus(Operator):
    def create(self):
        self.matrix[1, 0] = 1