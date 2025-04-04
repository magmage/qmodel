'''
qmodel - library for Hilbert space objects and simple quantum models
(CC0) Markus Penz
created on 01.08.2024

version 0.2.0 for supporting nested tensor products
current version 0.2.3 (14.11.2024)
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
    """
    Represents a Hilbert space basis.
    Basis elements are dictionaries that contain quantum numbers (qn).
    """

    def __init__(self, qn_key: str = None, qn_range: Iterable[int|float]|None = None):
        """
        Initializes the Basis.

        Parameters:
            qn_key (Optional[str]): The key for the quantum number.
            qn_range (Optional[Iterable[int, float]]): The range of quantum numbers.

        Raises:
            TypeError: If qn_key is not a string or qn_range is not an iterable of numbers.
        """
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
        """
        Returns a string representation of the Basis.
        """ 
        out = 'Basis: dim = {}, part num = {}, symmetry = {}, qn keys = {}\n'.format(self.dim, self.part_num, self.symmetry, self.qn_keys)
        out += 'Basis elements:\n'
        for e in self.el: ## or loop component bases
            out += str(e) + '\n'
        return out
    
    def tensor(self, basis: 'Basis') -> 'Basis':
        """
        Computes the tensor product of the current Basis with another Basis.

        Parameters:
            basis (Basis): Another Basis instance to tensor with.

        Returns:
            Basis: A new Basis instance representing the tensor product.

        Raises:
            TypeError: If the argument is not a Basis instance.
            ValueError: If the quantum numbers of the two bases are not disjoint.
        """       
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
        """
        Computes the n-fold tensor product of the basis with itself, applying the specified symmetry.

        Parameters:
            n (int): The number of times to tensor the basis with itself.
            symmetry (str): The symmetry type ('x' for none, 's' for symmetric, 'a' for antisymmetric).

        Returns:
            Basis: The resulting basis after tensoring and applying symmetry.

        Raises:
            ValueError: If n is not a non-negative integer or if symmetry is invalid.
        """        

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
        """
        Alias for antisymmetric tensor product space.

        Parameters:
            n (int): The number of times to tensor the basis.

        Returns:
            Basis: The resulting antisymmetric tensor product basis.
        """
        return self.ntensor(n, symmetry = 'a')

    def _find_qn(self, qn: dict|str) -> list:
        """
        Recursively finds quantum numbers in all elements and returns a list of results.

        Parameters:
            qn (dict | str): The quantum number(s) to search for.

        Returns:
            list: A list of dictionaries containing the search results.
        """
        # find qn recursively in all elements and return list of results
        # qn allows also multiple qn when its a dict like {'qn1': 1, 'qn2': 2}
        # either full qn dict is matched or just a single qn key is looked up
        # returns the full index, and then per qn_key the position in the basis-tree as a list and the permutation sign for antisymmetric bases
        res = []
        
        # subroutine for recursive call, starts with type(el) == tuple
        def _find_qn_el(qn: dict|str, el: dict|tuple, pos_list: list|None = None, permutation_sign_prev: int = 1): # find in element
            if pos_list is None: pos_list = []
            res = []
            for pos,e in enumerate(el[1:]): # skip first element in tuple (symmetry marker)
                permutation_sign = (-1 if el[0]=='a' else 1)**pos * permutation_sign_prev # only if antisymmetric, include sign from previous iteration
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
                elif type(qn) is str and qn in e.keys(): # for diag()
                    found.append({'index': i, 'res': [e[qn]]}) # return just value
            
        return found
    
    def sum(self, func: Callable):
        """
        Iterates all basis elements e and sums over the return value of func(e).

        Parameters:
            func (Callable): The function to apply to each basis element.

        Returns:
            Any: The sum of the function applied to all basis elements.
        """ 
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
        """
        Creates and returns the identity operator for the current basis.

        Returns:
            Operator: An identity operator acting on the same basis as self.
        """
        out_op = Operator(self)
        out_op.matrix = np.identity(self.dim, dtype=np.complex128)
        return out_op
    
    def diag(self, qn_key: str|None = None) -> 'Operator': # diagonal qn operator
        """
        Creates a diagonal operator based on a specified quantum number.

        Parameters:
            qn_key (Optional[str]): The quantum number key to use. If None and the basis has only one quantum number, it defaults to that key.

        Returns:
            Operator: A new operator with diagonal elements corresponding to the quantum number values. In a many-particle basis the values of all single-particle basis elements are summed.

        Raises:
            ValueError: If qn_key is not provided and cannot be determined, or if it is not part of the basis.
        """
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
        """
        Creates a Hilbert one-body operator / hopping operator.

        Creates operator a_qn1^dag a_qn2 considering the symmetry of the basis.
        qn can include several qn_keys and their values.
        If only qn1 is given, qn2 = qn1 and we get the number operator (projector on qn).

        Parameters:
            qn1 (dict): The quantum number(s) for the creation operator.
            qn2 (dict|None = None)): The quantum number(s) for the annihilation operator.

        Returns:
            Operator: The hopping operator.

        Raises:
            TypeError: If qn1 or qn2 are not dictionaries.
            ValueError: If the qn_keys in qn1 and qn2 do not match.
        """
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
    """
    A class representing a single-particle basis for lattice systems.
    """    
    def __init__(self, vertex_num: int, qn_key: str = 'i'):
        if int(vertex_num)!=vertex_num or vertex_num <= 0: raise ValueError('Vertex number must be positive and integer valued.')
        super().__init__( qn_key=qn_key, qn_range=range(1,vertex_num+1))

    def graph_laplacian(self, graph_edges: set, qn_key: str = 'i', hopping: float = 1, include_degree: bool = True) -> 'Operator':
        """
        Constructs the graph Laplacian operator for the lattice.

        Parameters:
            graph_edges (set): Set of edges representing the graph. Each edge is a tuple (i, j).
            qn_key (str): Quantum number key representing the site index.
            hopping (float): Hopping parameter (typically negative for electrons).
            include_degree (bool): Whether to include the degree term on the diagonal.

        Returns:
            Operator: The graph Laplacian operator.
        """   
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
        """
        Creates the potential operator based on the provided potential dictionary.

        Parameters:
            pot (dict): Dictionary mapping quantum numbers to potential values.
            qn_key (str): The quantum number key. Defaults to 'i'.

        Returns:
            Operator: The resulting potential operator.

        Raises:
            ValueError: If the quantum number key is not in the basis.
        """
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
    """
    Represents a vacuum basis (just C space).

    This class defines the vacuum state in a quantum system, where no particles exist.
    The dimension of the vacuum basis is 1, corresponding to the empty state.

    Attributes:
        dim (int): The dimension of the vacuum basis, always set to 1.
        qn_key (str): The key for the quantum number, representing the vacuum state (default: 'vac').
        qn_range (range): The range for the quantum number, set to range(0, 1) to represent the vacuum.
    """    
    def __init__(self, qn_key: str = 'vac'):
        super().__init__( qn_key=qn_key, qn_range=range(0, 1) )
        
class NumberBasis(Basis): # for single state boson Fock space
    """
    Represents a number basis for a single-state boson Fock space.

    This class defines a number basis for a bosonic system, where the quantum number 
    corresponds to the number of bosons. It also provides methods to construct creation, 
    annihilation, and position-like operators in the Fock space.
    """

    def __init__(self, max_num: int, qn_key: str = 'n'):
        if int(max_num)!=max_num or max_num <= 0: raise ValueError('Maximal number must be positive and integer valued.')
        super().__init__( qn_key=qn_key, qn_range=range(0, max_num) )
        """
        Initializes the NumberBasis.

        Parameters:
            max_num (int): The maximum number of bosons (number of the particle in the system)in the Fock space.
            qn_key (str): The key for the quantum number, representing the number of bosons.

        Raises:
            ValueError: If max_num is not a positive integer.
        """        
    def creator(self) -> 'Operator':
        """
        Generates the creation operator for the Fock space.

        The creation operator increases the number of bosons by one. The matrix representation
        is a lower diagonal matrix where the non-zero entries correspond to the square root
        of the next quantum number (number of bosons).

        Returns:
            Operator: The creation operator represented as a matrix.
        """
        # generate matrix / only -1 secondary-diagonal in qn
        out_op = Operator(self)
        if self.dim > 1:
            out_op.matrix = np.diag(np.sqrt(range(1,self.dim)), -1)
        else:
            out_op.matrix = np.identity(1, dtype=np.complex128)
        return out_op

    def annihilator(self) -> 'Operator':
        """
        Generates the annihilation operator for the Fock space.

        The annihilation operator decreases the number of bosons by one. The matrix representation
        is an upper diagonal matrix where the non-zero entries correspond to the square root
        of the current quantum number.

        Returns:
            Operator: The annihilation operator represented as a matrix.
        """ 
        # generate matrix / only +1 secondary-diagonal in qn
        out_op = Operator(self)
        out_op.matrix = np.diag(np.sqrt(range(1,self.dim)), +1)
        return out_op
    
    def x_operator(self) -> 'Operator':
        """
        Generates the position-like operator for the Fock space.

        The position operator is defined as the sum of the creation and annihilation operators
        divided by the square root of 2.

        Returns:
            Operator: Return the operator for the displacement coordinate represented as a matrix.
        """  
        return (self.creator() + self.annihilator()) / sqrt(2)
    
    def dx_operator(self) -> 'Operator':
        """
        Generates the momentum-like operator for the Fock space.

        The momentum operator is defined as the difference of the creation and annihilation
        operators divided by the square root of 2.

        Returns:
            Operator: Return the operator for the derivative w.r.t. the displacement coordinate represented as a matrix.
        """  
        return (-self.creator() + self.annihilator()) / sqrt(2)

class SpinBasis(Basis): # always single-particle
    """
    Represents a single-particle spin basis.

    This class defines the spin basis for a quantum system with two spin states (up and down).
    It provides methods to construct key spin operators, including the Pauli matrices and
    raising/lowering operators.
    """
    def __init__(self, qn_key: str = 's'):
        super().__init__( qn_key=qn_key, qn_range=(+1,-1))
        """
        Initializes the SpinBasis.

        Parameters:
            qn_key (str): The key for the quantum number, representing the spin states.

        Behavior:
            Initializes the spin basis with two states: spin-up (+1) and spin-down (-1).
        """    
        
    def sigma_x(self) -> 'Operator':
        """
        Generates the Pauli sigma-x operator.

        The sigma-x operator corresponds to a quantum spin flip operation, represented by:
        [[0, 1],
        [1, 0]]

        Returns:
            Operator: The sigma-x operator represented as a matrix.
        """ 
        out_op = Operator(self)
        out_op.matrix[0, 1] = 1
        out_op.matrix[1, 0] = 1
        return out_op

    def sigma_y(self) -> 'Operator':
        """
        Generates the Pauli sigma-y operator.

        The sigma-y operator corresponds to a spin operation in the y-direction, represented by:
        [[0, -i],
        [i, 0]]

        Returns:
            Operator: The sigma-y operator represented as a matrix.
        """    
        out_op = Operator(self)
        out_op.matrix[0, 1] = -1j
        out_op.matrix[1, 0] = 1j
        return out_op
        
    def sigma_z(self) -> 'Operator':
        """
        Generates the Pauli sigma-z operator.

        The sigma-z operator corresponds to a spin measurement along the z-axis, represented by:
        [[1, 0],
        [0, -1]]

        Returns:
            Operator: The sigma-z operator represented as a matrix.
        """           
        out_op = Operator(self)
        out_op.matrix[0, 0] = 1
        out_op.matrix[1, 1] = -1
        return out_op
        
    def sigma_vec(self) -> 'OperatorList':
        """
        Returns the vector of the three Pauli matrices (sigma_x, sigma_y, sigma_z).

        Returns:
            OperatorList: A list containing the sigma-x, sigma-y, and sigma-z operators.
        """          
        out_op = OperatorList(self, [self.sigma_x(), self.sigma_y(), self.sigma_z()])
        return out_op
        
    def sigma_plus(self) -> 'Operator':
        """
        Generates the spin raising (sigma-plus) operator.

        The sigma-plus operator raises the spin from down to up, represented by:
        [[0, 1],
        [0, 0]]

        Returns:
            Operator: The spin raising operator represented as a matrix.
        """          
        out_op = Operator(self)
        out_op.matrix[0, 1] = 1
        return out_op

    def sigma_minus(self) -> 'Operator':
        """
        Generates the spin lowering (sigma-minus) operator.

        The sigma-minus operator lowers the spin from up to down, represented by:
        [[0, 0],
        [1, 0]]

        Returns:
            Operator: The spin lowering operator represented as a matrix.
        """ 
        out_op = Operator(self)
        out_op.matrix[1, 0] = 1
        return out_op

    def proj_plus(self) -> 'Operator':
        """
        Generates the spin projection operator for the spin-up state.

        The projection operator selects the spin-up state, represented by:
        [[1, 0],
        [0, 0]]

        Returns:
            Operator: The projection operator for the spin-up state represented as a matrix.
        """
        out_op = Operator(self)
        out_op.matrix[0, 0] = 1
        return out_op

    def proj_minus(self) -> 'Operator':
        """
        Generates the spin projection operator for the spin-down state.

        The projection operator selects the spin-down state, represented by:
        [[0, 0],
        [0, 1]]

        Returns:
            Operator: The projection operator for the spin-down state represented as a matrix.
        """
        out_op = Operator(self)
        out_op.matrix[1, 1] = 1
        return out_op
        
class Vector:
    """
    Represents a vector in a given basis.

    This class defines a vector in a quantum mechanical basis and provides methods
    for performing vector operations such as multiplication, inner products, calculating
    the probability distribution, norm, and trace over a subbasis.

    Attributes:
        basis (Basis): The basis in which the vector is defined.
        col (np.ndarray): The column vector represented as a NumPy array or list.
    """
    # vector in a given basis
    def __init__(self, basis: Basis, col: list|np.ndarray = None): # can be initialized with column
        """
        Initializes the Vector.

        Parameters:
            basis (Basis): The quantum mechanical basis for the vector.
            col (list | np.ndarray, optional): The column vector, defaults to a zero vector if not provided.

        Raises:
            ValueError: If the shape of the column vector does not match the basis dimension.
        """
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
        """
        Returns a copy of the current vector.

        The method creates a new Vector instance with the same basis and a copy of the column vector.

        Returns:
            Vector: A copy of the current vector.
        """
        # return a copy of itself
        X = Vector(self.basis)
        X.col = self.col.copy()
        return X

    def _test_basis(self, X): # private method for basis comparison / use name as 'forward reference'
        """
        Private method to check if the basis of two vectors match.

        Parameters:
            X (Vector): The vector to compare with.

        Raises:
            TypeError: If the two vectors do not share the same basis.
        """   
        if self.basis != X.basis:
            raise TypeError('The bases do not match between the vectors.')

    def __neg__(self):
        """
        Negates the current vector.

        Returns:
            Vector: A new vector representing the negative of the current vector.
        """
        X = self.copy()
        X.col = -X.col
        return X
              
    def __add__(self, X):
        """
        Adds another vector to the current vector.

        Parameters:
            X (Vector): The vector to add.

        Returns:
            Vector: A new vector representing the sum.

        Raises:
            TypeError: If the addition is not valid.
        """
        Y = Vector(self.basis)
        if isinstance(X, Vector):
            # must have same basis and size
            self._test_basis(X)
            Y.col = np.add(self.col, X.col)         
        else:
            raise TypeError('Type error in addition.')
        
        return X
    
    def __sub__(self, X):
        """
        Subtracts another vector from the current vector.

        Parameters:
            X (Vector): The vector to subtract.

        Returns:
            Vector: A new vector representing the difference.
        """ 
        return self.__add__(-X)

    def __mul__(self, a):
        """
        Multiplies the vector by a scalar value.

        Parameters:
            a (numbers.Number): The scalar value to multiply the vector by.

        Returns:
            Vector: The result of multiplying the vector by the scalar.

        Raises:
            TypeError: If the input is not a scalar value.
        """        
        # mul operator, return new object X = self * a
        if not isinstance(a, numbers.Number):
            raise TypeError('Type error in multiplication.')        
        X = Vector(self.basis)
        X.col = a*self.col
        return X
    
    def __rmul__(self, a):
        """
        Right multiplication of the vector by a scalar value (commutative).

        Parameters:
            a (numbers.Number): The scalar value to multiply the vector by.

        Returns:
            Vector: The result of multiplying the vector by the scalar.
        """        
        # mul operator, return new object X = a * self
        # commutative in all cases
        # Operator * Vector in handled in Operator class
        return self.__mul__(a)
    
    def __truediv__(self, a):
        if not isinstance(a, numbers.Number):
            raise TypeError('Number type expected in division.')
        return self.__mul__(1/a)
    
    def inner(self, X) -> complex:
        """
        Computes the inner product of the current vector with another vector.

        The inner product is the complex conjugate of self times the second vector X.

        Parameters:
            X (Vector): The vector to compute the inner product with.

        Returns:
            complex: The inner product result.

        Raises:
            TypeError: If the input is not a vector.
        """   
        # inner product with second vector where self gets complex conjugated
        if not isinstance(X, Vector):
            raise TypeError('Inner product is only allowed with another vector.')
        self._test_basis(X)
        return np.dot(self.col.conj(), X.col)

    def prob(self) -> np.ndarray:
        """
        Returns the probability distribution for all basis elements.

        The probability is calculated as the absolute square of the vector's components.

        Returns:
            np.ndarray: The probability distribution for each basis element.
        """  
        # return |psi|^2 for all basis elements
        prob = np.zeros(self.basis.dim)
        for index,_ in enumerate(self.basis.el):
            prob[index] = abs(self.col[index])**2
        return prob
    
    def norm(self) -> float: # 2-norm of vector
        """
        Computes the 2-norm (Euclidean norm) of the vector.

        Returns:
            float: The 2-norm of the vector.
        """   
        return np.linalg.norm(self.col, ord=2)
            
    def trace(self, subbasis: Basis) -> np.ndarray:
        """
        Traces out all basis elements except for the subbasis and gives the sum of absolute squares.

        The trace operation sums over the probabilities of the basis elements.

        Parameters:
            subbasis (Basis): The subbasis to trace over.

        Returns:
            np.ndarray: The probability distribution for the subbasis.

        Raises:
            ValueError: If the subbasis is not primitive.
        """ 
        # trace out all basis elements except of subbasis and give sum abs-square
        # subbasis must be a primitive basis with only dict elements
        if subbasis.symmetry is not None: raise ValueError('This method only works on primitive bases.')
        prob = np.zeros(subbasis.dim)
        for index,e in enumerate(subbasis.el):
            prob[index] = self.basis.hop(e).expval(self, transform_real=True)
        return prob

    def __str__(self) -> str:
        """
        Provides a string representation of the vector.

        Returns:
            str: A string showing the vector's dimension and components.
        """
        out = 'Vector: dim = {}\n'.format(self.basis.dim)
        # and just show column
        return out + str(self.col)

class Operator:
    """
    Represents a quantum mechanical operator with respect to a given basis.
    
    This class provides a set of methods to perform operator algebra, 
    including basic arithmetic operations, eigenvalue computations, 
    commutators, and tensor products.
    
    Attributes:
        basis (Basis): The basis in which the operator is defined.
        matrix (np.ndarray): The matrix representation of the operator.
    """
    
    def __init__(self, basis: Basis):
        """
        Initializes the operator with respect to the given basis.

        Parameters:
            basis (Basis): The basis in which the operator is defined.

        Raises:
            TypeError: If the basis is not of type 'Basis'.
        """

        if not isinstance(basis, Basis):
            raise TypeError('Operator must be initialized with a basis.')
        
        self.basis = basis        
        # init zero matrix and create
        self.matrix = np.zeros((self.basis.dim, self.basis.dim), dtype=np.complex128)

    def copy(self) -> 'Operator':
        """
        Returns a copy of the current operator.

        Returns:
            Operator: A copy of the current operator.
        """ 
        # return a copy of itself
        A = Operator(self.basis)
        A.matrix = self.matrix.copy()
        return A

    def _test_basis(self, A): # private method for basis comparison / use name as 'forward reference'
        """
        Checks if the basis of the current operator matches the basis of another operator.

        Parameters:
            A (Operator): The operator to compare with.

        Raises:
            TypeError: If the bases of the two operators do not match.
        """  
        if self.basis != A.basis:
            raise TypeError('The bases do not match between the operators.')
    
    def __neg__(self):
        """
        Negates the current operator.

        Returns:
            Operator: A new operator representing the negative of the current operator.
        """
        B = self.copy()
        B.matrix = -B.matrix
        return B
              
    def __add__(self, A):
        """
        Adds another operator or a scalar to the current operator.

        Parameters:
            A (Operator or numbers.Number): The operator or scalar to add.

        Returns:
            Operator: A new operator representing the sum.

        Raises:
            TypeError: If the addition is not valid.
        """
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
        """
        Right-side addition, allowing scalar or operator addition to be commutative.

        Parameters:
            A (Operator or numbers.Number): The operator or scalar to add.

        Returns:
            Operator: A new operator representing the sum.
        """   
        return self.__add__(A) # is commutative
    
    def __sub__(self, A):
        """
        Subtracts another operator or scalar from the current operator.

        Parameters:
            A (Operator or numbers.Number): The operator or scalar to subtract.

        Returns:
            Operator: A new operator representing the difference.
        """ 
        return self.__add__(-A)
    
    def __rsub__(self, A):
        """
        Right-side subtraction, allowing scalar or operator subtraction to be commutative.

        Parameters:
            A (Operator or numbers.Number): The operator or scalar to subtract from.

        Returns:
            Operator: A new operator representing the difference.
        """   
        return -self.__sub__(A)
        
    def __mul__(self, A):
        """
        Multiplies the current operator by another operator, vector, scalar, or a list of scalars.

        Parameters:
            A (Operator, Vector, numbers.Number, or list): The entity to multiply with.

        Returns:
            Operator, Vector, or OperatorList: The result of the multiplication.

        Raises:
            TypeError: If the input type is not supported.
        """
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
        """
        Right-side multiplication, allowing scalar or operator multiplication.

        Parameters:
            A (Operator or numbers.Number): The operator or scalar to multiply with.

        Returns:
            Operator: The result of the multiplication.
        """
        if isinstance(A, Operator):
            # must have same basis and size
            self._test_basis(A)
            B = Operator(self.basis)
            B.matrix = np.matmul(A.matrix, self.matrix)
        else: # commutative in other cases
            B = self.__mul__(A)

        return B  
        
    def __truediv__(self, a):
        """
        Divides the operator by a scalar.

        Parameters:
            a (numbers.Number): The scalar to divide by.

        Returns:
            Operator: The resulting operator after division.

        Raises:
            TypeError: If the input is not a scalar.
        """
        if not isinstance(a, numbers.Number):
            raise TypeError('Number type expected in division.')
        B = Operator(self.basis)
        B.matrix = self.matrix/a
        return B
        
    def __pow__(self, a):
        """
        Raises the operator to the power of an integer.

        Parameters:
            a (int): The power to raise the operator to.

        Returns:
            Operator: The resulting operator after exponentiation.

        Raises:
            TypeError: If the input is not an integer.
        """
        if not isinstance(a, int):
            raise TypeError('Integer expected in power operation on operator.')
        B = Operator(self.basis)
        B.matrix = np.linalg.matrix_power(self.matrix, a)
        return B
    
    def conj(self) -> 'Operator': # conjugate
        """
        Returns the complex conjugate of the operator.

        Returns:
            Operator: The conjugate of the operator.
        """
        B = Operator(self.basis)
        B.matrix = self.matrix.conj()
        return B
    
    def T(self) -> 'Operator': # transpose
        """
        Returns the transpose of the operator.

        Returns:
            Operator: The transpose of the operator.
        """
        B = Operator(self.basis)
        B.matrix = self.matrix.transpose()
        return B
    
    def adj(self) -> 'Operator': # adjoint
        """
        Returns the adjoint (conjugate transpose) of the operator.

        Returns:
            Operator: The adjoint of the operator.
        """
        B = Operator(self.basis)
        B.matrix = self.matrix.transpose().conj()
        return B
    
    def add_adj(self) -> 'Operator': # add adjoint (h.c.) to operator itself
        """
        Adds the adjoint of the operator to itself (Hermitian conjugate).

        Returns:
            Operator: The resulting operator.
        """        
        B = Operator(self.basis)
        B.matrix = self.matrix + self.matrix.transpose().conj()
        return B
    
    def comm(self, A: 'Operator') -> 'Operator':
        """
        Computes the commutator [self, A] = self * A - A * self.

        Parameters:
            A (Operator): The operator to compute the commutator with.

        Returns:
            Operator: The resulting commutator.

        Raises:
            TypeError: If the input is not an operator.
        """
        if isinstance(A, Operator):
            # must have same basis and size (tested in mul anyway)
            return self*A - A*self
        else:
            raise TypeError('Type error in commutator.')
        
    def acomm(self, A: 'Operator') -> 'Operator':
        """
        Computes the anticommutator {self, A} = self * A + A * self.

        Parameters:
            A (Operator): The operator to compute the anticommutator with.

        Returns:
            Operator: The resulting anticommutator.

        Raises:
            TypeError: If the input is not an operator.
        """
        if isinstance(A, Operator):
            # must have same basis and size (tested in mul anyway)
            return self*A + A*self
        else:
            raise TypeError('Type error in anticommutator.')
    
    def expval(self, vec: Vector, check_real: bool = False, transform_real: bool = False) -> float|complex:
        """
        Calculates the expectation value of the operator with respect to a given vector.

        Parameters:
            vec (Vector): The vector to calculate the expectation value with.
            check_real (bool, optional): If True, checks that the result is real-valued.
            transform_real (bool, optional): If True, transforms the result to a real value.

        Returns:
            float or complex: The expectation value.

        Raises:
            TypeError: If the vector basis does not match the operator basis.
            ValueError: If the result is expected to be real but contains an imaginary component.
        """
        if self.basis != vec.basis:
            raise TypeError('The bases do not match between operator and vector.')
        
        val = np.dot(vec.col.conjugate(), np.dot(self.matrix, vec.col))
        if check_real and abs(val.imag) > IMAG_PART_ACCURACY:
            raise ValueError('Eigenvalue has imaginary component.')
        
        if transform_real:
            val = val.real
            
        return val

    def norm(self) -> float: # 2-norm of operator
        """
        Calculates the 2-norm of the operator.

        Returns:
            float: The 2-norm of the operator.
        """
        return np.linalg.norm(self.matrix, ord=2)
    
    def eig(self, hermitian: bool = False) -> dict:
        """
        Solves for the eigenvalues and eigenvectors of the operator.

        Parameters:
            hermitian (bool, optional): If True, assumes the operator is Hermitian.

        Returns:
            dict: A dictionary with eigenvalues, eigenvectors, and (if applicable) degeneracy.
        """
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
        """
        Tests if the operator is Hermitian (self-adjoint).

        Raises:
            ValueError: If the operator is not Hermitian.
        """
        if not np.allclose(self.matrix, self.matrix.conj().T, 0, HERMITICITY_ACCURACY):
            raise ValueError('Operator is not hermitian (self-adjoint).')

    def diag(self) -> dict:
        """
        Diagonalizes the operator (assumes Hermitian).
        The method first checks for self-adjointness, then returns the diagonal of the diagonalized operator \hat{D}
        and the corresponding transformation matrix \hat{U}.
        as NumPy ndarray in a dictionary with keys 'diag' and 'transform' . The relation to the operator is \hat{D} = \hat{U}^\dagger \hat{A} \hat{U}.

        Returns:
            dict: A dictionary with diagonal elements and the transformation matrix.

        Raises:
            ValueError: If the operator is not Hermitian.
        """
        # test if hermitian
        self.test_hermiticity()
        
        # solve for eigensystem
        res = np.linalg.eigh(self.matrix)
        d = res[0] # diagonal of matrix U.conj().T @ A @ U
        U = res[1] # eigenvectors
        return {'diag': d, 'transform': U.conj().T} # return numpy matrices, not Operators, because they are wrt to another basis 

    def extend(self, ext_basis: Basis) -> 'Operator':
        """
        Extends the operator to a larger basis in tensor space (block form).

        Parameters:
            ext_basis (Basis): The larger basis to extend the operator to.

        Returns:
            Operator: The extended operator.

        Raises:
            TypeError: If ext_basis is not of type Basis.
            ValueError: If the extended basis does not include the quantum numbers of the current basis.
        """
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
        """
        Computes the tensor product with another operator or operator list.

        Parameters:
            A (Union[Operator, OperatorList]): The operator or list to tensor with.
            tensor_basis (Basis, optional): The basis to use for the tensor product.

        Returns:
            Union[Operator, OperatorList]: The resulting tensor product.

        Raises:
            TypeError: If the second argument is not of type Basis.
        """
        # Note: should be able to also work with symmetric/antisymmetric basis
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
        """
        Returns a string representation of the operator.
        """
        out = 'Operator: dim = {}\n'.format(self.basis.dim)
        # and just show matrix
        return out + str(self.matrix)
    
    def __len__(self):
        """
        Returns the number of operators, which is always 1 for this class.
        """
        return 1

class OperatorList: # list of operators
    """
    Represents a list of quantum mechanical operators with respect to a given basis.

    This class provides functionality for working with lists of operators, including
    conjugation, adjoints, tensor products, and more advanced operations like
    simultaneous diagonalization.

    Attributes:
        basis (Basis): The basis in which the operators are defined.
        operators (list): A list of operators of type Operator.
    """
    def __init__(self, basis: Basis, operators: list|None = None): # do NOT use empty list as default! causes unwanted default value
        """
        Initializes the OperatorList with a basis and an optional list of operators.

        Parameters:
            basis (Basis): The basis in which the operators are defined.
            operators (list, optional): A list of Operator objects.

        Raises:
            TypeError: If the basis is not of type Basis, or if the operators are not of type list or Operator.
        """
        if not isinstance(basis, Basis):
            raise TypeError('OperatorList must be initialized with a basis.')
        if operators is not None and (type(operators) is not list or not all(isinstance(A, Operator) for A in operators)):
            raise TypeError('If the OperatorList is initialized with operators, those need to be a list of Operator-type objects.')
        
        self.basis = basis
        if operators is None: operators = [] # init with empty list
        self.operators = operators # init with arg

    def __getitem__(self, index) -> Operator: # make subscriptable
        """
        Enables indexing to retrieve an operator from the list.

        Parameters:
            index (int): The index of the operator to retrieve.

        Returns:
            Operator: The operator at the specified index.

        Raises:
            ValueError: If the index is not an integer or is out of range.
        """
        if not isinstance(index, int):
            raise ValueError('Index must be integer.')
        if index < 0 or index >= len(self.operators):
            raise ValueError('Index is out of range.')
        return self.operators[index]
    
    def append(self, operators: Operator|list):
        """
        Appends one or multiple operators to the OperatorList. Note that this method does not create a new instance.

        Parameters:
            operators (Operator or list): One or more operators to append.

        Raises:
            ValueError: If the appended operator(s) do not have the same basis as the OperatorList.
        """
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
        """
        Returns a copy of the current OperatorList.

        Returns:
            OperatorList: A new OperatorList that is a copy of the current one.
        """
        A = OperatorList(self.basis)
        for op in self.operators:
            A.append(op.copy())
        return A
    
    def __neg__(self):
        """
        Negates all operators in the OperatorList.

        Returns:
            OperatorList: A new OperatorList with all operators negated.
        """
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(-op) # makes copy
        return B
                    
    def __add__(self, A):
        """
        Adds two OperatorLists or adds a scalar to all operators in the list.

        Parameters:
            A (OperatorList or numbers.Number): The OperatorList or scalar to add.

        Returns:
            OperatorList: A new OperatorList that is the sum of the current and added objects.

        Raises:
            ValueError: If the dimensions of the OperatorLists do not match.
            TypeError: If the addition is performed with an invalid type.
        """
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
        """
        Multiplies the OperatorList element-wise by another list, scalar, or OperatorList.

        Parameters:
            A (list, np.ndarray, OperatorList, or numbers.Number): The object to multiply with.

        Returns:
            Operator or OperatorList: The result of the multiplication.

        Raises:
            ValueError: If the dimensions do not match for inner product.
            TypeError: If the multiplication is performed with an invalid type.
        """
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
        """
        Right multiplication, allows multiplication of scalar or list with OperatorList.

        Parameters:
            A (list, np.ndarray, or numbers.Number): The object to multiply with.

        Returns:
            Operator or OperatorList: The result of the right multiplication.
        """
        return self.__mul__(A) # is commutative
        
    def __truediv__(self, a):
        """
        Divides all operators in the list by a scalar.

        Parameters:
            a (numbers.Number): The scalar to divide by.

        Returns:
            OperatorList: A new OperatorList with operators divided by the scalar.

        Raises:
            TypeError: If the divisor is not a scalar.
        """
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
        """
        Returns the transpose of each operator in the list.

        Returns:
            OperatorList: A new OperatorList with the transpose of each operator.
        """
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op.T())
        return B

    def adj(self) -> 'OperatorList': # adjoint
        """
        Return an operator list with each operator Hermitian adjoined (complex conjugated and transposed).

        Returns:
            OperatorList: A new OperatorList with the adjoint of each operator.
        """
        B = OperatorList(self.basis)
        for op in self.operators:
            B.append(op.adj())
        return B
    
    def expval(self, vec: Vector, check_real = False, transform_real = False) -> float|complex:
        """
        Returns the expectation value of each operator with respect to a vector.

        Parameters:
            vec (Vector): The vector with respect to which the expectation value is computed.
            check_real (bool): If True, raises an error if the result is not real.
            transform_real (bool): If True, forces the result to be real.

        Returns:
            np.ndarray: An array of expectation values for each operator.
        """
        out = []
        for op in self.operators:
            out.append(op.expval(vec, check_real, transform_real))
        return np.array(out) # make it np array so one can add etc.

    def norm(self) -> float: # 2-norm of operator list
        """
        Return the Euclidean norm of the list of operators \sqrt{\sum_i \left\| \hat{A_i} \right\|^2}.

        Returns:
            float: The 2-norm of the operator list.
        """
        out = 0 
        for op in self.operators:
            out += op.norm()**2
        return sqrt(out)
    
    def sum(self) -> Operator: # sum and return Operator
        """
        Sums all operators in the list and returns the result as a single operator.

        Returns:
            Operator: A new operator that is the sum of all operators in the list.
        """
        B = Operator(self.basis)
        B.matrix = np.sum([A.matrix for A in self.operators], axis=0)
        return B
    
    def diag(self, trials: int = 5) -> dict:
        """
        Performs simultaneous diagonalization of hermitian operators.

        Parameters:
            trials (int): The number of random trials to attempt diagonalization.

        Returns:
            dict: A dictionary with the diagonal elements and transformation matrix.

        Raises:
            ValueError: If the operators cannot be diagonalized simultaneously.
        """
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
        """
        Extends the operator list to a larger basis in tensor space.

        Parameters:
            ext_basis (Basis): The basis to extend the operators into.

        Returns:
            OperatorList: A new OperatorList extended to the new basis.
        """
        out_op_list = OperatorList(ext_basis)
        for op in self.operators:
            out_op_list.append(op.extend(ext_basis))
        return out_op_list
    
    def tensor(self, A: 'Operator', tensor_basis: Basis = None) -> 'OperatorList': # return new OperatorList
        """
        Returns the tensor product of each operator in the list with another operator.

        Parameters:
            A (Operator): The operator to take the tensor product with.
            tensor_basis (Basis, optional): The basis in which to perform the tensor product.

        Returns:
            OperatorList: A new OperatorList that is the tensor product of the operators.
        
        Raises:
            TypeError: If A is not of type Operator.
        """
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
        """
        Returns a string representation of the OperatorList.

        Returns:
            str: A string representing the OperatorList.
        """
        s = 'OperatorList: len = {}\n'.format(len(self.operators))
        for i,op in enumerate(self.operators):
            s = s + str(i) + ':\n' + op.__str__() + '\n'
        return s
    
    def __len__(self):
        """
        Returns the number of operators in the list.

        Returns:
            int: The number of operators in the list.
        """    
        return len(self.operators)

