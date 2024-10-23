'''
qmodel - library for Hilbert space objects and simple quantum models
(CC0) Markus Penz
created on 01.08.2024

version 0.2.0 for supporting nested tensor products
current version 0.2.2 (23.10.2024)
'''

import itertools
import numpy as np
import numbers
from math import sqrt
from typing import Callable, Union, Iterable
from copy import copy
from scipy.linalg.lapack import cggev

# constants
DEGENERACY_ACCURACY = 1e-10
IMAG_PART_ACCURACY = 1e-14
HERMITICITY_ACCURACY = 1e-6 # absolute tolerance in allclose, also use for diagonal matrix (cannot be that accurate for simultaneous diagonalization)

# classes
class Basis:
    # Hilbert space basis
    # basis elements are dicts that contain quantum numbers (qn)

    def __init__(self, qn_key: str = None, qn_range: Iterable[int|float]|None = None):
        self.dim = 0
        self.qn_keys = set()
        self.part_num = 0 # components in tuple elements
        self.part_basis = self # component basis
        self.symmetry = None # None | 'x' | 's' | 'a'
        # symmetry = None has el = list of dicts {'qn_key1': qn1, 'qn_key2': qn2}
        # symmetry = 'x' | 's' | 'a' has el = nested tuples, length is part_num
        self.el = []
        
        # can be initialized as a primitive basis with a single qn (string) and its range (list of numbers)
        if not qn_key is None and not qn_range is None:
            if not isinstance(qn_key, str): raise TypeError('The qn key must be string valued.')
            self.part_num = 1
            self.qn_keys = {qn_key}
            if isinstance(qn_range, Iterable):
                self.dim = len(qn_range)
                for q in qn_range:
                    if isinstance(q, int) or isinstance(q, float):
                        self.el.append( {qn_key: q} )
                    else:
                        raise TypeError('The qn range must be an integer or float valued iterable.')
            else:
                raise TypeError('The qn range must be an integer or float valued iterable.')
            
    def __str__(self) -> str:
        out = 'Basis: dim = {}, part num = {}, symmetry = {}, qn keys = {}\n'.format(self.dim, self.part_num, self.symmetry, self.qn_keys)
        out += 'Basis elements:\n'
        for e in self.el: ## or loop component bases
            out += str(e) + '\n'
        return out
    
    def tensor(self, basis: 'Basis') -> 'Basis':
        # tensor product with another basis
        # cannot be used for tensor^n of same basis -> use ntensor
        if not isinstance(basis, Basis): raise TypeError('Argument must be of type Basis.')
        # bases must have disjoint qn
        if not self.qn_keys.isdisjoint(basis.qn_keys):
            raise ValueError('Quantum numbers in tensor product must be disjoint.')
        
        out_basis = Basis()
        out_basis.dim = self.dim * basis.dim
        out_basis.qn_keys = self.qn_keys.union(basis.qn_keys)
        
        # get elements as product
        el_tensor = itertools.product(self.el, basis.el)
        
        # convert to primitive basis with only dicts?
        if self.symmetry is None and basis.symmetry is None:
            # elements as merged dicts
            out_basis.el = [{**t[0], **t[1]} for t in el_tensor]
        else:
            # elements as tuples but still part_num = 1
            # save symmetry marker 'x'
            out_basis.symmetry = 'x'
            out_basis.part_num = 2
            out_basis.el = [('x',) + t for t in el_tensor]
        
        return out_basis
    
    def ntensor(self, n: int, symmetry: str = 'x') -> 'Basis':
        # n-fold tensor product
        if int(n)!=n or n < 0: raise ValueError('n must be non-negative and integer valued.')
        if not symmetry in ['x', 's', 'a']: raise ValueError('Symmetry must be of type \'x\', \'s\', or \'a\'')
        if n == 0: return VacuumBasis()
        if n == 1: return copy(self) # nothing to do
        
        out_basis = Basis()
        out_basis.qn_keys = self.qn_keys # same qn keys
        out_basis.symmetry = symmetry
        
        # if not nested because bases share same symmetry type then remove symmetry marker before combining them
        if symmetry != self.symmetry: # nested
            el_self = self.el
            out_basis.part_num = n
            out_basis.part_basis = self
        else:
            el_self = [e[1:] for e in self.el]
            out_basis.part_num = self.part_num * n
            out_basis.part_basis = self.part_basis
        
        # always perform full tensor product and sort our later
        el_tensor = itertools.product(el_self, repeat=n)
        
        # if not nested, flatten elements
        if symmetry == self.symmetry:
            el_tensor = [tuple(itertools.chain.from_iterable(e)) for e in el_tensor]
            
        # add symmetry marker and sort out wrong elements for 's' and 'a'
        el_final = []
        for e in el_tensor:
            # check symmetry
            inc = True # include this element
            if symmetry in ['s', 'a']:
                pos = 0
                for e_part in out_basis.part_basis.el: # must be in that order
                    if symmetry == 's':
                        while pos < out_basis.part_num and e[pos] == e_part: # can have multiple e_part
                            pos += 1
                        if pos == out_basis.part_num:
                            break
                    elif symmetry == 'a':
                        if e[pos] == e_part: # only single e_part
                            pos += 1
                        if pos == out_basis.part_num:
                            break
                inc = (pos == out_basis.part_num) # all found in right order
            if inc: # include this element?
                # add symmetry marker
                el_final.append( (symmetry,) + e )
        
        out_basis.dim = len(el_final)
        out_basis.el = el_final
        return out_basis

    def wedge(self, n: int) -> 'Basis':
        # alias for antisymmetric tensor product space
        return self.ntensor(n, symmetry = 'a')

    def _find_qn(self, qn: dict|str, pos_list: list|None = None, permutation_sign: int = 1) -> list:
        # find qn recursively in all elements and return list of results
        # qn allows also multiple qn when its a dict like {'qn1': 1, 'qn2': 2}
        # either full qn dict is matched or just a single qn key is looked up
        # returns the full index, and then per qn_key the position in the basis-tree as a list and the permutation sign for antisymmetric bases
        res = []
        if pos_list is None: pos_list = []
        
        # subroutine for recursive call, starts with type(el) == tuple
        def _find_qn_el(qn: dict|str, el: dict|tuple, pos_list: list|None = None, permutation_sign: int = 1): # find in element
            if pos_list is None: pos_list = []
            res = []
            for pos,e in enumerate(el[1:]): # skip first element in tuple (symmetry marker)
                permutation_sign = (-1 if el[0]=='a' else 1)**pos * permutation_sign # only if antisymmetric
                if type(e) is dict:
                    if type(qn) is dict and {key:e[key] for key in e if key in qn.keys()} == qn: # qn subset of e
                        # return basis and where it was found with permutation_sign
                        res.append({'found_pos': [*pos_list, pos], 'permutation_sign': permutation_sign})
                    elif type(qn) is str and qn in e.keys(): # for diag()
                        res.append(e[qn]) # return just value
                elif type(e) is tuple:
                    res.extend(_find_qn_el(qn, e, [*pos_list, pos], permutation_sign)) # recursive call

            return res
        
        found = []
        for i,e in enumerate(self.el):
            if type(e) is tuple:
                res = _find_qn_el(qn, e)
                if res:
                    found.append({'index': i, 'res': res})
            elif type(e) is dict:
                if type(qn) is dict and {key:e[key] for key in e if key in qn.keys()} == qn: # compare directly in primitive basis, qn subset of e
                    found.append({'index': i, 'res': [{'found_pos': [], 'permutation_sign': 1}]})
                elif type(qn) is str and qn in e.keys():
                    found.append({'index': i, 'res': [e[qn]]}) # return just value
            
        return found
    
    def sum(self, func: Callable) -> float:
        # iterate all basis elements e and sum over return value of func(e)
        # could sum over operators or vectors
        re = None
        for e in self.el:
            if re is None:
                re = func(e)
            else:
                re += func(e)
        return re
    
    def id(self) -> 'Operator': # identity operator
        out_op = Operator(self)
        out_op.matrix = np.identity(self.dim, dtype=np.complex128)
        return out_op
    
    def diag(self, qn_key: str|None = None) -> 'Operator': # diagonal qn operator
        # basic operator is just diagonal x -> x for a quantum number
        # qn_key can be omitted if basis just has one qn
        # if found multiple times in many-particle state then added
        if qn_key is None and len(self.qn_keys) == 1: qn_key = next(iter(self.qn_keys)) # default if there is only one
        elif not qn_key in self.qn_keys: raise ValueError('Quantum number is not a part of the basis.')
        out_op = Operator(self)
        
        found = self._find_qn(qn_key)
        for f in found:
            out_op.matrix[f['index'], f['index']] = sum(f['res'])
            
        return out_op
    
    def hop(self, qn1: dict, qn2: dict|None = None) -> 'Operator': # Hilbert one-body operator / hopping operator
        # create operator a_qn1^dag a_qn2 regarding symmetry of basis
        # qn can include several qn_keys and their values
        # if only qn1 is given qn2 = qn1 and we get the number operator == projector on qn (if it is basis el)
        if qn2 is None: qn2 = qn1
        
        out_op = Operator(self)
        
        # qn1 and qn2 must be dicts
        if not isinstance(qn1, dict) or not isinstance(qn2, dict):
            raise TypeError('The qn must be given as dicts.')    
        # qn_keys in qn1 and qn2 must agree (comparing View objects)
        if qn1.keys() != qn2.keys():
            raise ValueError('The two qn must have the same keys.')
        
        # subroutine for recursive call
        def _remove_qn_el(el: dict|tuple, pos_list: list, pos_current: list|None = None, symmetry_current: str|None = None): # remove qn at given position from dict
            if pos_current is None: pos_current = []
            res = []
            dict_removed = None
            
            if type(el) is dict:
                if pos_list != pos_current:
                    res = [el]
                else: # at correct position
                    dict_removed = el
                    if symmetry_current == 'x': # mark deleted element for tensor basis (not symmetric) 
                        res = [None]
            elif type(el) is tuple:
                for i,e in enumerate(el[1:]):
                    res_app, dict_removed_new = _remove_qn_el(e, pos_list, [*pos_current, i], el[0])
                    if dict_removed_new is not None: dict_removed = dict_removed_new
                    res.extend(res_app)
                    
            return [res,
                    dict_removed]
        
        found1 = self._find_qn(qn1)
        if qn1 == qn2:
            found2 = found1 # no need to search again
        else:
            found2 = self._find_qn(qn2)
        
        for f1 in found1:
            e1 = self.el[f1['index']]
            for f2 in found2:
                e2 = self.el[f2['index']]
                # loop all results in basis elements (qn can be found multiple times)
                for r1 in f1['res']:
                    e1r,d1 = _remove_qn_el(e1, r1['found_pos'])
                    for r2 in f2['res']:
                        e2r,d2 = _remove_qn_el(e2, r2['found_pos'])
                        # basis elements after removal of dict AND dicts after removal of searched keys have to agree
                        if e1r == e2r and \
                        {key1:d1[key1] for key1 in d1 if key1 not in qn1} == {key2:d2[key2] for key2 in d2 if key2 not in qn2}:
                            out_op.matrix[f1['index'],f2['index']] += r1['permutation_sign']*r2['permutation_sign'] # permutation_sign for antisymmetric basis
        
        return out_op

class LatticeBasis(Basis): # always single-particle
    def __init__(self, vertex_num: int, qn_key: str = 'i'):
        if int(vertex_num)!=vertex_num or vertex_num <= 0: raise ValueError('Vertex number must be positive and integer valued.')
        super().__init__( qn_key=qn_key, qn_range=range(1,vertex_num+1))

    def graph_laplacian(self, graph_edges: set, qn_key: str = 'i', hopping: float = 1, include_degree: bool = True) -> 'Operator':
        # (Def. 6 in Graph-DFT paper)
        # hopping(real) * a_i^dag a_j for i != j and -deg(i) on diagonal (0 if include_degree==False)
        # graph_edges = {(1,2), (2,3), (1,3)} (triangle)
        # w.r.t. given qn_key
        if not qn_key in self.qn_keys: raise ValueError('Quantum number is not a part of the basis.')
        # create Operator
        out_op = Operator(self)
        # count degree
        deg = dict()
        # loop edges
        for edge in graph_edges:
            if len(edge) != 2: raise ValueError('Graph edge must be given with length 2.')
            fr = edge[0]
            to = edge[1]
            # add to degree
            if include_degree:
                deg[fr] = deg.get(fr, 0) + 1
                deg[to] = deg.get(to, 0) + 1
            # all qn that have qn_key = fr
            for e1 in self.part_basis.el:
                if e1[qn_key]==fr:
                    # make same qn with qn_key = to for connection with onebody_operator (first only one directions)
                    e2 = e1.copy()
                    e2[qn_key] = to
                    out_op += self.hop(e1, e2)
        # multiply hopping
        out_op *= hopping
        # add transpose (is real)
        out_op += out_op.T()
        # set minus degree to diagonal
        if include_degree:
            out_op -= self.graph_pot(deg, qn_key)
        return out_op

    def graph_pot(self, pot: dict|list|np.ndarray, qn_key: str = 'i') -> 'Operator':
        # v_i * a_i^dag a_i
        # pot = {1: 1, 2: -1, 3: 0} or [1,-1,0]
        # w.r.t. given qn_key
        if not qn_key in self.qn_keys: raise ValueError('Quantum number is not a part of the basis.')
        if not isinstance(pot, dict) and not isinstance(pot, list) and not isinstance(pot, np.ndarray):
            raise TypeError('Potential must be of dict, list or NumPy ndarray type.')
        # create Operator
        out_op = Operator(self)
        for e in self.part_basis.el:
            if isinstance(pot, dict) and e[qn_key] in pot:
                out_op += pot[e[qn_key]] * self.hop(e)
            elif (isinstance(pot, list) or isinstance(pot, np.ndarray)) and e[qn_key] <= len(pot):
                out_op += pot[e[qn_key]-1] * self.hop(e)
        return out_op

class VacuumBasis(Basis): # just C space
    def __init__(self, qn_key: str = 'vac'):
        super().__init__( qn_key=qn_key, qn_range=range(0, 1) )
        
class NumberBasis(Basis): # for single state boson Fock space
    def __init__(self, max_num: int, qn_key: str = 'n'):
        if int(max_num)!=max_num or max_num <= 0: raise ValueError('Maximal number must be positive and integer valued.')
        super().__init__( qn_key=qn_key, qn_range=range(0, max_num) )
        
    def creator(self) -> 'Operator':
        # generate matrix / only -1 secondary-diagonal in qn
        out_op = Operator(self)
        if self.dim > 1:
            out_op.matrix = np.diag(np.sqrt(range(1,self.dim)), -1)
        else:
            out_op.matrix = np.identity(1, dtype=np.complex128)
        return out_op

    def annihilator(self) -> 'Operator':
        # generate matrix / only +1 secondary-diagonal in qn
        out_op = Operator(self)
        out_op.matrix = np.diag(np.sqrt(range(1,self.dim)), +1)
        return out_op
    
    def x_operator(self) -> 'Operator':
        return (self.creator() + self.annihilator()) / sqrt(2)
    
    def dx_operator(self) -> 'Operator':
        return (-self.creator() + self.annihilator()) / sqrt(2)

class SpinBasis(Basis): # always single-particle
    def __init__(self, qn_key: str = 's'):
        super().__init__( qn_key=qn_key, qn_range=(+1,-1))
        
    def sigma_x(self) -> 'Operator':
        out_op = Operator(self)
        out_op.matrix[0, 1] = 1
        out_op.matrix[1, 0] = 1
        return out_op

    def sigma_y(self) -> 'Operator':
        out_op = Operator(self)
        out_op.matrix[0, 1] = -1j
        out_op.matrix[1, 0] = 1j
        return out_op
        
    def sigma_z(self) -> 'Operator':
        out_op = Operator(self)
        out_op.matrix[0, 0] = 1
        out_op.matrix[1, 1] = -1
        return out_op
        
    def sigma_vec(self) -> 'OperatorList':
        out_op = OperatorList(self, [self.sigma_x(), self.sigma_y(), self.sigma_z()])
        return out_op
        
    def sigma_plus(self) -> 'Operator':
        out_op = Operator(self)
        out_op.matrix[0, 1] = 1
        return out_op

    def sigma_minus(self) -> 'Operator':
        out_op = Operator(self)
        out_op.matrix[1, 0] = 1
        return out_op

    def proj_plus(self) -> 'Operator':
        out_op = Operator(self)
        out_op.matrix[0, 0] = 1
        return out_op

    def proj_minus(self) -> 'Operator':
        out_op = Operator(self)
        out_op.matrix[1, 1] = 1
        return out_op
        
class Vector:
    # vector in a given basis
    def __init__(self, basis: Basis, col: list|np.ndarray = None): # can be initialized with column
        self.basis = basis
        if col is None:
            self.col = np.zeros(basis.dim, dtype=np.complex128)
        else:
            # check if size of array and basis match
            col = np.array(col)
            if col.shape != (basis.dim,):
                raise ValueError('Vector shape does not match basis dimension.')
            self.col = col

    def copy(self) -> 'Vector':
        # return a copy of itself
        X = Vector(self.basis)
        X.col = self.col.copy()
        return X

    def _test_basis(self, X): # private method for basis comparison / use name as 'forward reference'
        if self.basis != X.basis:
            raise TypeError('The bases do not match between the vectors.')
        
    def __mul__(self, a):
        # mul operator, return new object X = self * a
        if not isinstance(a, numbers.Number):
            raise TypeError('Type error in multiplication.')        
        X = Vector(self.basis)
        X.col = a*self.col
        return X
    
    def __rmul__(self, a):
        # mul operator, return new object X = a * self
        # commutative in all cases
        # Operator * Vector in handled in Operator class
        return self.__mul__(a)
    
    def __truediv__(self, a):
        if not isinstance(a, numbers.Number):
            raise TypeError('Number type expected in division.')
        return self.__mul__(1/a)
    
    def inner(self, X) -> complex:
        # inner product with second vector where self gets complex conjugated
        if not isinstance(X, Vector):
            raise TypeError('Inner product is only allowed with another vector.')
        self._test_basis(X)
        return np.dot(self.col.conj(), X.col)

    def prob(self) -> np.ndarray:
        # return |psi|^2 for all basis elements
        prob = np.zeros(self.basis.dim)
        for index,_ in enumerate(self.basis.el):
            prob[index] = abs(self.col[index])**2
        return prob
    
    def norm(self) -> float: # 2-norm of vector
        return np.linalg.norm(self.col, ord=2)
            
    def trace(self, subbasis: Basis) -> np.ndarray:
        # trace out all basis elements except of subbasis and give sum abs-square
        # subbasis must be a primitive basis with only dict elements
        if subbasis.symmetry is not None: raise ValueError('This method only works on primitive bases.')
        prob = np.zeros(subbasis.dim)
        for index,e in enumerate(subbasis.el):
            prob[index] = self.basis.hop(e).expval(self, transform_real=True)
        return prob

    def __str__(self) -> str:
        out = 'Vector: dim = {}\n'.format(self.basis.dim)
        # and just show column
        return out + str(self.col)

class Operator:
    # matrix w.r.t to basis
    
    def __init__(self, basis: Basis):
        if not isinstance(basis, Basis):
            raise TypeError('Operator must be initialized with a basis.')
        
        self.basis = basis        
        # init zero matrix and create
        self.matrix = np.zeros((self.basis.dim, self.basis.dim), dtype=np.complex128)

    def copy(self) -> 'Operator':
        # return a copy of itself
        A = Operator(self.basis)
        A.matrix = self.matrix.copy()
        return A

    def _test_basis(self, A): # private method for basis comparison / use name as 'forward reference'
        if self.basis != A.basis:
            raise TypeError('The bases do not match between the operators.')
    
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
            self._test_basis(A)
            B.matrix = np.add(self.matrix, A.matrix)         
        elif isinstance(A, numbers.Number):
            B.matrix = self.matrix + A*np.identity(self.basis.dim, dtype=np.complex128)
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
            self._test_basis(A)
            B = Operator(self.basis)
            B.matrix = np.matmul(self.matrix, A.matrix)
        elif isinstance(A, Vector):
            # must have same basis and size
            self._test_basis(A)
            B = Vector(self.basis)
            B.col = np.dot(self.matrix, A.col)
        elif isinstance(A, numbers.Number):
            B = Operator(self.basis)
            B.matrix = A*self.matrix
        elif isinstance(A, list):
            # return OperatorList
            B_vec = OperatorList(self.basis)
            for a in A:
                if isinstance(a, numbers.Number):
                    B_vec.operators.append(self * a)
                else:
                    raise TypeError('Type error in multiplication with a list.')
            return B_vec
        else:
            raise TypeError('Type error in multiplication.')

        return B        
    
    def __rmul__(self, A):
        # mul operator, return new object B = A * self
        # also handels scalar multiplication
        if isinstance(A, Operator):
            # must have same basis and size
            self._test_basis(A)
            B = Operator(self.basis)
            B.matrix = np.matmul(A.matrix, self.matrix)
        else: # commutative in other cases
            B = self.__mul__(A)

        return B  
        
    def __truediv__(self, a):
        if not isinstance(a, numbers.Number):
            raise TypeError('Number type expected in division.')
        B = Operator(self.basis)
        B.matrix = self.matrix/a
        return B
        
    def __pow__(self, a):
        if not isinstance(a, int):
            raise TypeError('Integer expected in power operation on operator.')
        B = Operator(self.basis)
        B.matrix = np.linalg.matrix_power(self.matrix, a)
        return B
    
    def conj(self) -> 'Operator': # conjugate
        B = Operator(self.basis)
        B.matrix = self.matrix.conj()
        return B
    
    def T(self) -> 'Operator': # transpose
        B = Operator(self.basis)
        B.matrix = self.matrix.transpose()
        return B
    
    def adj(self) -> 'Operator': # adjoint
        B = Operator(self.basis)
        B.matrix = self.matrix.transpose().conj()
        return B
    
    def add_adj(self) -> 'Operator': # add adjoint (h.c.) to operator itself
        B = Operator(self.basis)
        B.matrix = self.matrix + self.matrix.transpose().conj()
        return B
    
    def comm(self, A: 'Operator') -> 'Operator':
        if isinstance(A, Operator):
            # must have same basis and size (tested in mul anyway)
            return self*A - A*self
        else:
            raise TypeError('Type error in commutator.')
        
    def acomm(self, A: 'Operator') -> 'Operator':
        if isinstance(A, Operator):
            # must have same basis and size (tested in mul anyway)
            return self*A + A*self
        else:
            raise TypeError('Type error in anticommutator.')
    
    def expval(self, vec: Vector, check_real: bool = False, transform_real: bool = False) -> float|complex:
        if self.basis != vec.basis:
            raise TypeError('The bases do not match between operator and vector.')
        
        val = np.dot(vec.col.conjugate(), np.dot(self.matrix, vec.col))
        if check_real and abs(val.imag) > IMAG_PART_ACCURACY:
            raise ValueError('Eigenvalue has imaginary component.')
        
        if transform_real:
            val = val.real
            
        return val

    def norm(self) -> float: # 2-norm of operator
        return np.linalg.norm(self.matrix, ord=2)
    
    def eig(self, hermitian: bool = False) -> dict:
        # solve for eigensystem
        if hermitian: # trust the input, is not checked
            res = np.linalg.eigh(self.matrix) # eigenvalues ordered, lowest first
            # count degeneracy in order of eigenvalues
            deg = []
            index = 0
            while index < len(res[0]):
                count = np.count_nonzero(np.abs(res[0][index] - res[0]) < DEGENERACY_ACCURACY)
                deg.append(count)
                # jump over all items with same value
                index += count
            deg_dict = {'degeneracy': deg}
        else:
            res = np.linalg.eig(self.matrix)
            deg_dict = {}
            
        return {
                'eigenvalues': res[0],
                'eigenvectors': [Vector(self.basis, v) for v in np.transpose(res[1])], # eigenvectors are cols
                **deg_dict
            }
        
    def test_hermiticity(self):
        # test if it is really hermitian conjugate
        if not np.allclose(self.matrix, self.matrix.conj().T, 0, HERMITICITY_ACCURACY):
            raise ValueError('Operator is not hermitian (self-adjoint).')

    def diag(self) -> dict:
        # test if hermitian
        self.test_hermiticity()
        
        # solve for eigensystem
        res = np.linalg.eigh(self.matrix)
        d = res[0] # diagonal of matrix U.conj().T @ A @ U
        U = res[1] # eigenvectors
        return {'diag': d, 'transform': U.conj().T} # return numpy matrices, not Operators, because they are wrt to another basis 

    def extend(self, ext_basis: Basis) -> 'Operator':
        # extend to a larger basis in tensor space (block form)
        if not isinstance(ext_basis, Basis): raise TypeError('Argument must be of type Basis.')
        # qn keys have to match
        if len(self.basis.qn_keys.intersection(ext_basis.qn_keys)) != len(self.basis.qn_keys):
            raise ValueError('The extended basis does not include the quantum numbers of the previous operator basis.')
        
        out_op = Operator(ext_basis)
        
        for index1,qn1 in enumerate(self.basis.part_basis.el):
            for index2,qn2 in enumerate(self.basis.part_basis.el):
                if self.matrix[index1, index2] != 0:
                    out_op += ext_basis.hop(qn1, qn2) * self.matrix[index1, index2]
                        
        return out_op
    
    def tensor(self, A: Union['Operator', 'OperatorList'], tensor_basis: Basis = None) -> Union['Operator', 'OperatorList']: # return new Operator or OperatorList
        # tensor basis, can also be provided to allow creation of tensorized operators on same basis 
        if tensor_basis is None:
            tensor_basis = self.basis.tensor(A.basis)
        elif not isinstance(tensor_basis, Basis):
            raise TypeError('Second (optional) argument must be of type Basis.')
            
        if isinstance(A, Operator):
            # new Operator
            out_op = Operator(tensor_basis)
            out_op.matrix = np.kron(self.matrix, A.matrix)
            return out_op
        elif isinstance(A, OperatorList): # apply on whole list
            out_op_list = OperatorList(tensor_basis)
            for op in A.operators:
                out_op_list.append(self.tensor(op, tensor_basis))
            return out_op_list
        else:
            raise TypeError('Type error in tensor.')
        
    def __str__(self):
        out = 'Operator: dim = {}\n'.format(self.basis.dim)
        # and just show matrix
        return out + str(self.matrix)
    
    def __len__(self):
        return 1

class OperatorList: # list of operators
    def __init__(self, basis: Basis, operators: list|None = None): # do NOT use empty list as default! causes unwanted default value
        if not isinstance(basis, Basis):
            raise TypeError('OperatorList must be initialized with a basis.')
        if operators is not None and (type(operators) is not list or not all(isinstance(A, Operator) for A in operators)):
            raise TypeError('If the OperatorList is initialized with operators, those need to be a list of Operator-type objects.')
        
        self.basis = basis
        if operators is None: operators = [] # init with empty list
        self.operators = operators # init with arg

    def __getitem__(self, index) -> Operator: # make subscriptable
        if not isinstance(index, int):
            raise ValueError('Index must be integer.')
        if index < 0 or index >= len(self.operators):
            raise ValueError('Index is out of range.')
        return self.operators[index]
    
    def append(self, operators: Operator|list):
        # append one or multiple operators
        # careful, this does not create a new instance
        if isinstance(operators, list):
            for op in operators:
                self.append(op) # recursive call
        elif isinstance(operators, Operator):
            # test is same basis
            if self.basis != operators.basis:
                raise ValueError("Appended operators must have the same basis as OperatorList.")
    
            self.operators.append(operators)
        else:
            raise ValueError('Type for append must be list or Operator.')

    def copy(self) -> 'OperatorList':
        # return a copy of itself
        A = OperatorList(self.basis)
        for op in self.operators:
            A.append(op.copy())
        return A
    
    def __neg__(self):
        # negation, return new object B = -A
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(-op) # makes copy
        return B
                    
    def __add__(self, A):
        # add operator vector, return new object B = self + A
        # also handles scalar addition
        B = OperatorList(self.basis)
        if isinstance(A, OperatorList):
            # must have same size
            if len(self.operators) != len(A.operators):
                raise ValueError('Operator vector dimensions do not agree.')
            for i,op in enumerate(self.operators):
                B.append(op + A.operators[i])
        elif isinstance(A, numbers.Number):
            for op in self.operators:
                B.append(op + A)
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
            for i,op in enumerate(self.operators):
                B = B + op * A[i]
        elif isinstance(A, OperatorList): # inner product
            B = self.__mul__(A.operators) # call with list
        elif isinstance(A, numbers.Number):
            B = OperatorList(self.basis)
            for op in self.operators:
                B.append(op * A)
        else:
            raise TypeError('Type error in multiplication.')

        return B
    
    __array_priority__ = 10000 # use __rmul__ instead of NumPy's __mul__
    def __rmul__(self, A):
        return self.__mul__(A) # is commutative
        
    def __truediv__(self, a):
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op/a) # type is checked in Operator __truediv__
        return self
    
    def conj(self) -> 'OperatorList': # conjugate
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op.conj())
        return B
    
    def T(self) -> 'OperatorList': # transpose
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op.T())
        return B

    def adj(self) -> 'OperatorList': # adjoint
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op.adj())
        return B
    
    def expval(self, vec: Vector, check_real = False, transform_real = False) -> float|complex:
        out = []
        for op in self.operators:
            out.append(op.expval(vec, check_real, transform_real))
        return np.array(out) # make it np array so one can add etc.

    def norm(self) -> float: # 2-norm of operator list
        out = 0 
        for op in self.operators:
            out += op.norm()**2
        return sqrt(out)
    
    def sum(self) -> Operator: # sum and return Operator
        B = Operator(self.basis)
        B.matrix = np.sum([A.matrix for A in self.operators], axis=0)
        return B
    
    def diag(self, trials: int = 5) -> dict:
        # simultaneous diagonalization of hermitian matrices
        # returns diagonals and transformation matrix
        # inspired by randomized simultaneous diagonalization by congruence method
        # https://arxiv.org/html/2402.16557v1
        # https://github.com/haoze12345/rsdc/blob/main/rnojd.py
        for t in range(trials):
            w = np.random.normal(0,1,len(self))
            A1 = w*self # inner product
            A2 = self.sum()
            # use simultaneous diagonalization from LAPACK
            _, _, _, X, _, _ = cggev(A1.matrix,A2.matrix)
            Ua = X / np.linalg.norm(X,axis=0)
            U = Ua.conj().T
            # diagonalize all operators
            out = []
            for op in self.operators:
                M = U @ op.matrix @ Ua
                d = np.diag(M).real
                out.append(d)
                # test diag
                all_diag = True
                if not np.allclose(M, np.diag(d), 0, HERMITICITY_ACCURACY): # make it matrix again for comparison
                    all_diag = False
            if all_diag:
                return {'diag': out, 'transform': U} # return numpy matrices not Operators because is wrt to different basis
        if t+1==trials:
            raise ValueError('Operators could not be diagonalized simultaneously, maybe some are non-hermitian or they do not mutually commute.')
                            
    def extend(self, ext_basis: Basis) -> 'OperatorList':
        out_op_list = OperatorList(ext_basis)
        for op in self.operators:
            out_op_list.append(op.extend(ext_basis))
        return out_op_list
    
    def tensor(self, A: 'Operator', tensor_basis: Basis = None) -> 'OperatorList': # return new OperatorList
        # tensor basis, can also be provided to allow creation of tensorized operators on same basis
        # only between OperatorList and Operator, the other order is implemented with Operator
        # tensor order does not even matter since different qn keep track of that
        if tensor_basis is None:
            tensor_basis = self.basis.tensor(A.basis)
            
        if isinstance(A, Operator):
            # new OperatorList
            out_op_list = OperatorList(tensor_basis)
            for op in self.operators:
                out_op_list.append(op.tensor(A, tensor_basis))
            return out_op_list
        else:
            raise TypeError('Type error in tensor.')
        
    def __str__(self):
        s = 'OperatorList: len = {}\n'.format(len(self.operators))
        for i,op in enumerate(self.operators):
            s = s + str(i) + ':\n' + op.__str__() + '\n'
        return s
    
    def __len__(self):
        return len(self.operators)

