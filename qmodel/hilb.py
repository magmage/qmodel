## library for Hilbert space objects
## previously used for photon-free QED / Breit project
## taken over to Oslo QEDFT project / 2024, Feb 14
## removed grid- and Slater-basis parts / 2024, Feb 15
## operator vectors are replaced by general operator lists / 2024, Feb 15

## Wishlist 2024, Feb 19
# * Slater/fermion basis
# * basis change

import numpy as np
import re
import copy
import numbers
from collections.abc import Iterable

class BasisElement:
    # element from a basis
    # corresponding to an index and a dict of quatum numbers
    
    def __init__(self, index = None, qn = {}):
        self.index = index
        self.qn = qn # dict with qn
        
    def qn_sublist(self, qn_keys):
        # give list of qn selected by qn_keys
        return [self.qn[key] for key in qn_keys]
    
    def qn_get_unique(self):
        # if has just a unique qn get value
        if len(self.qn) != 1:
            raise ValueError('This method can only be called if basis is indexed by a single quantum number.')
        return next(iter(self.qn.values()))


class Basis:
    # indexed basis set
    # each basis element has a set of quantum numbers (qn)
    
    def __init__(self, *args): # init with arbitrary number of qn_keys, qn_ranges
        self.dim = 0 # number of basis elements
        self.qn_keys = set() # set of quantum numbers ({} is dict)
        self.qn_ranges = {}
        self.el = [] # list of BasisElements
        
        # args must be tuples of qn keys and their range
        for arg in args:
            if type(arg) != tuple or len(arg) != 2 or type(arg[0]) != str or not isinstance(arg[1], Iterable):
                raise TypeError('Arguments of the basis constructor must be tuples with two elements, the quantum number label and its range of indices.')
            self.extend(arg[0], arg[1])
    
    def extend(self, qn_key, qn_range): # extended basis object
        # create new basis
        ext_basis = Basis()
        index = 0
        for q in qn_range:
            ext_basis.el.append(BasisElement(index = index, qn = {qn_key: q}))
            index = index + 1
            
        ext_basis.dim = index
        ext_basis.qn_keys = {qn_key}
        ext_basis.qn_ranges = {qn_key: qn_range}
        
        if len(self.el) > 0:
            # tensor with existing basis
            self.__dict__ = self.tensor(ext_basis).__dict__
        else:
            # return as new
            self.__dict__ = ext_basis.__dict__

    def tensor(self, B: 'Basis'): # returns tensored basis / use name as 'forward reference'
        # tensor product with other basis
        # could be reference to the same basis, the copy
        if self is B: B = copy.deepcopy(B)
        
        # cannot have same labels for qn
        # todo: use [i] for dupliacte keys ?
        replace_B_qn = {}
        taken_keys = self.qn_keys.union(B.qn_keys) # those keys are all taken
        for qn_key in self.qn_keys.intersection(B.qn_keys):
            # add number to key to make it unique
            new_key = qn_key
            while new_key in taken_keys:
                res = re.findall('_([0-9]+)$', new_key)
                if len(res):
                    new_key = re.sub('_([0-9]+)$', '_' + str(int(res[0])+1), new_key)
                else:
                    new_key = qn_key + '_1'
            replace_B_qn[qn_key] = new_key
            taken_keys.add(new_key)
            
        # perform possible replacement
        B.replace_qn_keys(replace_B_qn)

        # create new basis
        tensor_basis = Basis()
        index = 0
        # loop both bases
        for self_e in self.el:
            for B_e in B.el:
                # append new basis element 
                tensor_basis.el.append(BasisElement(index = index, qn = {**self_e.qn, **B_e.qn}))
                index = index + 1
        
        tensor_basis.dim = index
        tensor_basis.qn_keys = self.qn_keys.union(B.qn_keys)
        tensor_basis.qn_ranges = {**self.qn_ranges, **B.qn_ranges}
        
        return tensor_basis
    
    def subbasis(self, select_keys: set):
        # select subbasis corresponding to given keys
        B = Basis()
        for key in self.qn_keys.intersection(select_keys): # must also be in present basis
            B.extend(key, self.qn_ranges[key])
        return B

    def list(self):
        for e in self.el:
            print(e.index, '->', e.qn)
        
    def find_el(self, qn: dict):
        # find all elements that fit qn
        # could be subset of qn
        qn_keys_intersection = list(self.qn_keys.intersection(qn.keys()))
        qn_search = {key:qn[key] for key in qn if key in qn_keys_intersection}
        return [el for el in self.el
            if {key:el.qn[key] for key in el.qn if key in qn_keys_intersection} == qn_search]
        
    def replace_qn_keys(self, replace: dict):
        for fr,to in replace.items():
            # change in keys set
            self.qn_keys.add(to)
            self.qn_keys.remove(fr)
            # change in ranges
            self.qn_ranges[to] = self.qn_ranges[fr]
            del self.qn_ranges[fr]
            # change in every element
            for e in self.el:
                e.qn[to] = e.qn[fr]
                del e.qn[fr]

            
class Vector:
    # vector in a given basis
    def __init__(self, basis: Basis, col = None):
        # check if size of array and basis match
        if col.shape != (basis.dim,):
            raise ValueError('Vector shape does not match basis dimension.')
        
        self.basis = basis
        if col is None:
            self.col = np.zeros(basis.dim, dtype=np.complex_)
        else:
            self.col = np.array(col)
        
    def prob(self):
        # give |psi|^2 for all basis elements
        for e in self.basis.el:
            print(str(e.qn) + ' -> ' + str(abs(self.col[e.index])**2))

    def trace_subbasis(self, subbasis: Basis):
        # trace out all basis elements except of subbasis and give sum abs-square
        prob = np.zeros(subbasis.dim)
        for e in subbasis.el:
            fs = self.basis.find_el(e.qn) # matching elements in big basis
            prob[e.index] = sum([abs(self.col[f.index])**2 for f in fs]) # sum abs squares
        return prob
    
    def trace(self, qn_keys: set):
        # trace out all basis elements *except* of passed qn_keys and give sum abs-square
        if not isinstance(qn_keys, set): qn_keys = {qn_keys} # make it a set
        subbasis = self.basis.subbasis(qn_keys) # make it a subbasis
        return self.trace_subbasis(subbasis)
                
    def filter(self, qn: dict):
        fs = self.basis.find_el(qn) # matching elements in big basis
        return np.array([self.col[f.index] for f in fs])
    
    def filter_abssq(self, qn: dict):
        return np.square(np.absolute(self.filter(qn)))

    def __str__(self):
        # just show column
        return str(self.col)

class Operator:
    # matrix w.r.t to basis
    # allow basis change?
    
    def __init__(self, basis: Basis):
        self.basis = basis        
        # init zero matrix and create
        self.matrix = np.zeros((self.basis.dim, self.basis.dim), dtype=np.complex_)
        self.create()
    
    def create(self):
        # do nothing here
        return
    
    def copy(self):
        # return a copy of itself
        A = Operator(self.basis)
        A.matrix = self.matrix.copy()
        return A
    
    def __test_basis(self, A: 'Operator'): # private method for basis comparison / use name as 'forward reference'
        if self.basis != A.basis:
            raise TypeError('The bases do not match between the operators to be added.')
    
    def __neg__(self):
        # negation, return new object B = -A
        B = self.copy()
        B.matrix = -B.matrix
        return B
              
    def __add__(self, A):
        # add operators, return new object B = self + A
        # also handels scalar addition
        B = Operator(self.basis)
        if isinstance(A, Operator):
            # must have same basis and size
            self.__test_basis(A)
            B.matrix = np.add(self.matrix, A.matrix)         
        elif isinstance(A, numbers.Number):
            B.matrix = self.matrix + A*np.identity(self.basis.dim, dtype=np.complex_)
        else:
            raise TypeError('Type error in addition.')
        
        return B
    
    def __radd__(self, A):
        return self.__add__(A) # is commutative
    
    def __sub__(self, A):
        return self.__add__(-A)
    
    def __rsub__(self, A):
        return -self.__sub__(A)
        
    def __mul__(self, A):
        # mul operator, return new object B = self * A
        # also handels scalar multiplication
        if isinstance(A, Operator):
            # must have same basis and size
            self.__test_basis(A)
            B = Operator(self.basis)
            B.matrix = np.matmul(self.matrix, A.matrix)            
        elif isinstance(A, numbers.Number):
            B = Operator(self.basis)
            B.matrix = A*self.matrix
        elif isinstance(A, list):
            # return OperatorList
            B_vec = OperatorList(self.basis)
            for a in A:
                B_vec.operators.append(self * a)
            return B_vec
        else:
            raise TypeError('Type error in multiplication.')

        return B        
    
    def __rmul__(self, A):
        # mul operator, return new object B = A * self
        # also handels scalar multiplication
        if isinstance(A, Operator):
            # must have same basis and size
            self.__test_basis(A)
            B = Operator(self.basis)
            B.matrix = np.matmul(A.matrix, self.matrix)
        else: # commutative in other cases
            B = self.__mul__(A)

        return B  
        
    def __truediv__(self, a):
        if not isinstance(a, numbers.Number):
            raise TypeError('Number type expected in multiplication from left.')
        B = Operator(self.basis)
        B.matrix = self.matrix/a
        return B
        
    def __pow__(self, a):
        if not isinstance(a, int):
            raise TypeError('Integer expected in power operation on operator.')
        B = Operator(self.basis)
        B.matrix = np.linalg.matrix_power(self.matrix, a)
        return B
    
    def conj(self): # conjugate
        B = Operator(self.basis)
        B.matrix = self.matrix.conj()
        return B

    def adj(self): # adjoint
        B = Operator(self.basis)
        B.matrix = self.matrix.transpose().conj()
        return B
    
    def comm(self, A):
        if isinstance(A, Operator):
            # must have same basis and size
            self.__test_basis(A)
            return self*A - A*self
        else:
            raise TypeError('Type error in commutator.')
        
    def acomm(self, A):
        if isinstance(A, Operator):
            # must have same basis and size
            self.__test_basis(A)
            return self*A + A*self
        else:
            raise TypeError('Type error in anticommutator.')
        
    def extend(self, ext_basis: Basis): # directly extend / return self reference
        # extend to a larger basis in tensor space (block form)
        # qn keys have to match
        if len(self.basis.qn_keys.intersection(ext_basis.qn_keys)) != len(self.basis.qn_keys):
            raise ValueError('The extended basis does not include the quantum numbers of the previous operator basis.')
        
        new_matrix = np.zeros((ext_basis.dim, ext_basis.dim), dtype=np.complex_)
        redundant_qn_keys = ext_basis.qn_keys.difference(self.basis.qn_keys)
        
        for ext_e1 in ext_basis.el:
            for ext_e2 in ext_basis.el:
                # if on diagonal of qn that are used in ext_basis but not in basis
                if ext_e1.qn_sublist(redundant_qn_keys) == ext_e2.qn_sublist(redundant_qn_keys):
                    # find corresponding elements
                    e1s = self.basis.find_el(ext_e1.qn)
                    e2s = self.basis.find_el(ext_e2.qn)
                    if len(e1s) != 1 or len(e2s) != 1:
                        raise RuntimeError('Assignement to the new basis failed, the corresponding basis element was not found.')
                    new_matrix[ext_e1.index, ext_e2.index] = self.matrix[e1s[0].index, e2s[0].index]
        
        self.basis = ext_basis
        self.matrix = new_matrix
        return self
    
    def tensor(self, A: 'Operator'): # return self reference
        # tensor basis
        new_basis = self.basis.tensor(A.basis)
        new_matrix = np.zeros((new_basis.dim, new_basis.dim), dtype=np.complex_)
        
        # loop through new basis
        for new_e1 in new_basis.el:
            for new_e2 in new_basis.el:
                # find corresponding basis elements in old bases
                self_e1s = self.basis.find_el(new_e1.qn)
                self_e2s = self.basis.find_el(new_e2.qn)
                A_e1s = A.basis.find_el(new_e1.qn)
                A_e2s = A.basis.find_el(new_e2.qn)
                if len(self_e1s) != 1 or len(self_e2s) != 1 or len(A_e1s) != 1 or len(A_e2s) != 1:
                    raise RuntimeError('Assignement to the new basis failed, the corresponding basis element was not found.')
                new_matrix[new_e1.index, new_e2.index] = self.matrix[self_e1s[0].index, self_e2s[0].index] * A.matrix[A_e1s[0].index, A_e2s[0].index]

        # change basis and matrix
        self.basis = new_basis
        self.matrix = new_matrix
        return self
    
    def expval(self, vec: Vector, check_real = False):
        if self.basis != vec.basis:
            raise TypeError('The bases do not match between the operators to be added.')
        
        val = np.dot(vec.col.conjugate().transpose(), np.dot(self.matrix, vec.col))
        if check_real:
            if abs(val.imag) > 1e-14:
                raise ValueError('Eigenvalue has imaginary component.')
            else:
                val = val.real # remove small imaginary part
                
        return val

    def test_hermiticity(self):
        # test if it is really hermitian conjugate
        err = np.linalg.norm(self.matrix - self.matrix.conj().T)
        if err > 1e-14:
            print('Hermiticity error: {:.1E}'.format(err))
            
    def __str__(self):
        # just show matrix
        return str(self.matrix)
    
    def __len__(self):
        return 1


class OperatorList: # list of operators
    def __init__(self, basis: Basis, operators = []):
        self.basis = basis
        self.operators = operators # init with arg

    def append(self, operators):
        # append one or multiple operators
        if isinstance(operators, list):
            for op in operators:
                self.append(op) # recursive call
        elif isinstance(operators, Operator):
            self.operators.append(operators)
        else:
            raise ValueError('Type for append must be list or Operator.')
    
    def copy(self):
        # return a copy of itself
        A = OperatorList(self.basis)
        for i in range(0, len(self.operators)):
            A.operators.append(self.operators[i].copy())
        return A

    def __test_len(self, A: 'OperatorList'): # private method for basis comparison / use name as 'forward reference'
        if len(self.operators) != len(A.operators):
            raise ValueError('Operator vector dimensions do not agree.')

    def __neg__(self):
        # negation, return new object B = -A
        B = self.copy()
        for i in range(0, len(B.operators)):
            B.operators[i].matrix = -B.operators[i].matrix
        return B
                    
    def __add__(self, A):
        # add operator vector, return new object B = self + A
        # also handels scalar addition
        B = OperatorList(self.basis)
        if isinstance(A, OperatorList):
            # must have same size
            self.__test_len(A)
            for i in range(0, len(self.operators)):
                B.operators.append(self.operators[i] + A.operators[i])
        elif isinstance(A, numbers.Number):
            for i in range(0, len(self.operators)):
                B.operators.append(self.operators[i] + A)
        else:
            raise TypeError('Type error in addition.')
        
        return B
    
    # handeled by parent:
    #def __radd__(self, A)
    #def __sub__(self, A)
    #def __rsub__(self, A)
        
    def __mul__(self, A):
        # mul operator vector, return new object B = self * A
        # with scalar or inner product with list
        if isinstance(A, list) or isinstance(A, np.ndarray):
            # must have same size
            if len(self.operators) != len(A):
                raise ValueError('Operator vector and list dimensions do not agree for inner product.')
            B = Operator(self.basis)
            for i in range(0, len(self.operators)):
                B = B + self.operators[i] * A[i]
        elif isinstance(A, numbers.Number):
            B = OperatorList(self.basis)
            for i in range(0, len(self.operators)):
                B.operators.append(self.operators[i] * A)
        else:
            raise TypeError('Type error in multiplication.')

        return B
    
    def __rmul__(self, A):
        return self.__mul__(A) # is commutative
        
    def __truediv__(self, a):
        if not isinstance(a, numbers.Number):
            raise TypeError('Number type expected in multiplication from left.')
        B = OperatorList(self.basis)
        for i in range(0, len(self.operators)):
            B.operators.append(self.operators[i]/a)
        return self
    
    def normsquare(self):
        # return Operator object
        B = Operator(self.basis) 
        for i in range(0, len(self.operators)):
            B.matrix = B.matrix + np.matmul(self.operators[i].matrix.conj(), self.operators[i].matrix)
        return B
    
    def conj(self): # conjugate
        B = OperatorList(self.basis)
        for i in range(0, len(self.operators)):
            B.operators.append(self.operators[i].conj())
        return B

    def adj(self): # adjoint
        B = OperatorList(self.basis)
        for i in range(0, len(self.operators)):
            B.operators.append(self.operators[i].adj())
        return B
    
    def extend(self, ext_basis: Basis):
        operators = []
        for op in self.operators:
            operators.append(op.extend(ext_basis))
        self.basis = ext_basis
        self.operators = operators
        return self
        
    def __str__(self):
        s = ''
        for i in range(0, len(self.operators)):
            s = s + str(i) + ':\n' + self.operators[i].__str__() + '\n'
        return s
    
    def __len__(self):
        return len(self.operators)


class qnOperator(Operator):
    # basic operator is just diagonal x -> x for a quantum number
    def __init__(self, basis, qn_key):
        if not qn_key in basis.qn_keys: raise ValueError('Quantum number is not a part of the given basis.')
        self.qn_key = qn_key # save
        super().__init__(basis)
        
    def create(self):
        for e in self.basis.el:
            self.matrix[e.index, e.index] = e.qn[self.qn_key]

                   
class Identity(Operator):
    def create(self):
        self.matrix = np.identity(self.basis.dim, dtype=np.complex_)
