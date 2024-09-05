'''
qmodel - library for Hilbert space objects and simple quantum models
(CC0) Markus Penz
created on 01.08.2024
new version 0.2.0 for supporting nested tensor products

future idea: Fock space by just adding basis elements (direct sum)
must adapt creation/annihilation
'''

import itertools
import numpy as np
import numbers
from math import sqrt
from typing import Callable, Union, Iterable
from copy import copy

# constants
DEGENERACY_ACCURACY = 1e-10
IMAG_PART_ACCURACY = 1e-14
HERMITICITY_ACCURACY = 1e-14

# classes
class Basis:
    # Hilbert space basis
    # basis elements are dicts that contain quantum numbers (qn)

    def __init__(self, qn_key: str = None, qn_range: Iterable[float]|None = None):
        self.dim = 0
        self.qn_keys = set()
        self.part_num = 1 # components in tuple elements
        self.part_basis = self # component basis
        self.symmetry = None # None | 'x' | 's' | 'a'
        # symmetry = None has el = list of dicts {'qn_key1': qn1, 'qn_key2': qn2}
        # symmetry = 'x' | 's' | 'a' has el = nested tuples, length is part_num
        self.el = []
        
        # can be initialized as a primitive basis with a single qn (string) and its range (list of numbers)
        if not qn_key is None and not qn_range is None:
            if not isinstance(qn_key, str): raise TypeError('The qn key must be string valued.')
            self.qn_keys = {qn_key}
            if isinstance(qn_range, Iterable):
                self.dim = len(qn_range)
                for q in qn_range:
                    self.el.append( {qn_key: q} )
            else:
                raise TypeError('The qn range must be a float valued iterable.')
            
    def __str__(self):
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
    
    def sum(self, func: Callable):
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
        out_op.matrix = np.identity(self.dim, dtype=np.complex_)
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

    ## must be on many-particle basis
    ## def fermion_graph_interaction(self, interaction_pot: dict, qn_key='i') -> 'Operator':
    
    def graph_pot(self, pot: dict, qn_key: str = 'i') -> 'Operator':
        # v_i * a_i^dag a_i
        # pot = {1: 1, 2: -1, 3: 0}
        # w.r.t. given qn_key
        if not qn_key in self.qn_keys: raise ValueError('Quantum number is not a part of the basis.')
        # create Operator
        out_op = Operator(self)
        for e in self.part_basis.el:
            if e[qn_key] in pot:
                out_op += pot[e[qn_key]] * self.hop(e)
        return out_op

class VacuumBasis(Basis): # just C space
    def __init__(self):
        super().__init__( qn_key='vac', qn_range=range(0, 1) )
        
class NumberBasis(Basis): # for single state boson Fock space
    def __init__(self, max_num = 2, qn_key = 'n'):
        super().__init__( qn_key=qn_key, qn_range=range(0, max_num) )
        
    def creator(self) -> 'Operator':
        # generate matrix / only -1 secondary-diagonal in qn
        out_op = Operator(self)
        if self.dim > 1:
            out_op.matrix = np.diag(np.sqrt(range(1,self.dim)), -1)
        else:
            out_op.matrix = np.identity(1, dtype=np.complex_)
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
    def __init__(self, qn_key = 's'):
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
    def __init__(self, basis: Basis, col = None): # can be initialized with column
        self.basis = basis
        if col is None:
            self.col = np.zeros(basis.dim, dtype=np.complex_)
        else:
            # check if size of array and basis match
            col = np.array(col)
            if col.shape != (basis.dim,):
                raise ValueError('Vector shape does not match basis dimension.')
            self.col = col
        
    def prob(self):
        # print |psi|^2 for all basis elements
        for index,e in enumerate(self.basis.el):
            print(str(e) + ' -> ' + str(abs(self.col[index])**2))
            
    def norm(self): # 2-norm of vector
        return np.linalg.norm(self.col, ord=2)
            
    def trace(self, subbasis: Basis):
        # trace out all basis elements except of subbasis and give sum abs-square
        # subbasis must be a primitive basis with only dict elements
        if subbasis.symmetry is not None: raise ValueError('This method only works on primitive bases.')
        prob = np.zeros(subbasis.dim)
        for index,e in enumerate(subbasis.el):
            prob[index] = self.basis.hop(e).expval(self, transform_real=True)
        return prob

    def __str__(self):
        out = 'Vector: dim = {}\n'.format(self.basis.dim)
        # and just show column
        return out + str(self.col)

class Operator:
    # matrix w.r.t to basis
    
    def __init__(self, basis: Basis):
        self.basis = basis        
        # init zero matrix and create
        self.matrix = np.zeros((self.basis.dim, self.basis.dim), dtype=np.complex_)

    def copy(self) -> 'Operator':
        # return a copy of itself
        A = Operator(self.basis)
        A.matrix = self.matrix.copy()
        return A

    def _test_basis(self, A): # private method for basis comparison / use name as 'forward reference'
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
            self._test_basis(A)
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
            self._test_basis(A)
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
    
    def T(self): # transpose
        B = Operator(self.basis)
        B.matrix = self.matrix.transpose()
        return B
    
    def adj(self): # adjoint
        B = Operator(self.basis)
        B.matrix = self.matrix.transpose().conj()
        return B
    
    def add_adj(self): # add adjoint (h.c.) to operator itself
        B = Operator(self.basis)
        B.matrix = self.matrix + self.matrix.transpose().conj()
        return B
    
    def norm(self): # 2-norm of operator
        return np.linalg.norm(self.matrix, ord=2)
    
    def comm(self, A):
        if isinstance(A, Operator):
            # must have same basis and size
            self._test_basis(A)
            return self*A - A*self
        else:
            raise TypeError('Type error in commutator.')
        
    def acomm(self, A):
        if isinstance(A, Operator):
            # must have same basis and size
            self._test_basis(A)
            return self*A + A*self
        else:
            raise TypeError('Type error in anticommutator.')
    
    def expval(self, vec: Vector, check_real = False, transform_real = False):
        if self.basis != vec.basis:
            raise TypeError('The bases do not match between the operators to be added.')
        
        val = np.dot(vec.col.conjugate(), np.dot(self.matrix, vec.col))
        if check_real and abs(val.imag) > IMAG_PART_ACCURACY:
            raise ValueError('Eigenvalue has imaginary component.')
        
        if transform_real:
            val = val.real
            
        return val

    def eig(self, hermitian = False):
        # solve for eigensystem
        if hermitian: # save with operator ?
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
        err = np.linalg.norm(self.matrix - self.matrix.conj().T)
        if err > HERMITICITY_ACCURACY:
            print('Hermiticity error: {:.1E}'.format(err))

    def extend(self, ext_basis: Basis): # returns new instance
        # extend to a larger basis in tensor space (block form)
        if not isinstance(ext_basis, Basis): raise TypeError('Argument must be of type Basis.')
        # qn keys have to match
        if len(self.basis.qn_keys.intersection(ext_basis.qn_keys)) != len(self.basis.qn_keys):
            raise ValueError('The extended basis does not include the quantum numbers of the previous operator basis.')
        
        out_op = Operator(ext_basis)
        
        for index1,qn1 in enumerate(self.basis.part_basis.el):
            for index2,qn2 in enumerate(self.basis.part_basis.el):
                if self.matrix[index1, index2] != 0: # optimize
                    out_op += ext_basis.hop(qn1, qn2) * self.matrix[index1, index2]
                        
        return out_op
    
    def tensor(self, A: Union['Operator', 'OperatorList'], tensor_basis: Basis = None) -> Union['Operator', 'OperatorList']: # return new Operator or OperatorList
        # tensor basis, can also be provided to allow creation of tensorized operators on same basis 
        if tensor_basis is None:
            tensor_basis = self.basis.tensor(A.basis)
            
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
    def __init__(self, basis: Basis, operators = None): # do NOT use empty list as default! causes unwanted default value
        self.basis = basis
        if operators is None: operators = [] # init with empty list
        self.operators = operators # init with arg
        ## check basis

    def append(self, operators):
        # append one or multiple operators
        if isinstance(operators, list):
            for op in operators:
                self.append(op) # recursive call
        elif isinstance(operators, Operator):
            self.operators.append(operators.copy()) # always make copy, not only reference
        else:
            raise ValueError('Type for append must be list or Operator.')
        ## check basis

    def copy(self):
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
    
    def __rmul__(self, A):
        return self.__mul__(A) # is commutative
        
    def __truediv__(self, a):
        if not isinstance(a, numbers.Number):
            raise TypeError('Number type expected in multiplication from left.')
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op/a)
        return self
    
    def conj(self): # conjugate
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op.conj())
        return B
    
    def T(self): # transpose
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op.T())
        return B

    def adj(self): # adjoint
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op.adj())
        return B

    def norm(self): # 2-norm of operator list
        out = 0 
        for op in self.operators:
            out += op.norm()**2
        return sqrt(out)
    
    def expval(self, vec: Vector, check_real = False, transform_real = False):
        out = []
        for op in self.operators:
            out.append(op.expval(vec, check_real, transform_real))
        return np.array(out) # make it np array so one can add etc.
    
    def extend(self, ext_basis: Basis): # return new instance
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

b = Basis('i', range(3))
print(b.hop({'i':1,'j':1}))