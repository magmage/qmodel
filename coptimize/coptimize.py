'''
Constrained Search in Imaginary Time main lib
@author: Markus Penz
'''
from qmodel import Operator, OperatorList, Vector
from math import pi
import numpy as np
import scipy
import random

def timestep(evol_matrix, x0, tau, method='Euler'):
    # timestep in evolution equation -d/dt x(t) = G(t).x(t) (includes minus)
    if method=='Euler':
        x = x0 - tau*(evol_matrix @ x0)
    elif method=='Exp':
        x = scipy.sparse.linalg.expm_multiply(-tau*evol_matrix, x0)
    elif method=='Crank-Nicolson':
        I = np.eye(*evol_matrix.shape)
        A = I + tau/2*evol_matrix
        b = np.dot(I - tau/2*evol_matrix, x0)
        x = np.linalg.solve(A, b)
    return x

def random_phase(Psi: np.ndarray, method: str):
    if method == 'phase':
        s = np.exp(1j*2*pi*np.random.uniform(low=0, high=1, size=Psi.shape[0]))
    elif method == 'sign': # just sign flip
        s = np.random.choice([1, -1], size=Psi.shape[0])
    else:
        raise ValueError("Unknown phase-change method: '{}'".format(method))
    
    return np.multiply(Psi, s)

class ConstraintOptimization:
    # method parameters
    ENERGY_CONVERGENCE_ACCURACY = 1e-6 # should be high, but slows down convergence
    IMPLICIT_CONVERGENCE_ACCURACY = 1e-8 # on Psi_sc vector, can be quite good
    ENERGY_CONSTRAINT_ACCURACY = 1e-2 # for expval(G) = 0 constraint, cannot be too good
    CONSTRAINT_ACCURACY = 1e-3 # on mu(Psi) = b constraint
    DELTA_TAU = 0.0005 # if too large constraints cannot be kept
    MIN_ITER_MAIN = 100 # min iterations in time (need some phase flips to reach gs); choose higher value together with RANDOM_PHASE_CHANCE if glitches from excited states appear
    MAX_ITER_MAIN = 20000 # max iterations in time
    MAX_ITER_SC = 200 # max iterations in self-consistent loop to determine beta at delta_tau/2, usually just a few needed
    RANDOM_PHASE_CHANCE = 1 # 1/this value is probability for random phase change, set to 0 for never; choose 1 or low if glitches from excited states appear; value=1 can even speed up convergence

    def __init__(self, A: Operator, B: OperatorList):
        # initialize with objective operator A and constraint operators B (id is added automatically)
        # A and all B hermitian
        # B all need to commute and be linearly independent (together with id)
        
        # test arguments    
        if not isinstance(A, Operator):
            raise TypeError("'A' must be of type qmodel.Operator.")
        if not isinstance(B, OperatorList):
            raise TypeError("'A' must be of type qmodel.OperatorList.")
        if A.basis != B.basis:
            raise ValueError("'A' and 'B' must have the same basis.")
        
        A.test_hermiticity()
        [B[i].test_hermiticity() for i in range(len(B))]
        
        # add id to B
        B_ = B.copy() # do not change passed B by appending below
        B_.append(B.basis.id()) # last item, not 0th item like in paper
        # perform simultaneous diagonalization of all matrices in B_
        res = B_.diag()
        # store matrices in basis where B are all diagonal
        self.dim = A.basis.dim
        self.num_constraints = len(B)+1 # M+1
        self.B_diag = res['diag'] # list of diagonalized B matrices (real)
        self.U = res['transform'] # transformation matrix
        self.Ua = self.U.conj().T # adjoint
        self.A_trans = self.U @ A.matrix @ self.Ua # A in new basis

    def solve_beta(self, Psi: np.ndarray):
        # solve for beta and also return generator
        # get Gram matrix GramB
        GramB = np.zeros((self.num_constraints,self.num_constraints)) # must be real symmetric in the setting of commuting B_i
        Psi2 = np.multiply(Psi, Psi.conj()) # element-wise |Psi|^2
        for i in range(self.num_constraints):
            Psi2Bi = np.multiply(Psi2, self.B_diag[i])
            for j in range(i+1): # j<=i
                GramB[i,j] = np.inner(self.B_diag[j], Psi2Bi).real
                if i!=j:
                    GramB[j,i] = GramB[i,j]
        
        # get r.h.s. for beta equation
        APsi2 = np.multiply(self.A_trans @ Psi, Psi.conj())
        gamma = np.zeros(self.num_constraints)
        for i in range(self.num_constraints):
            gamma[i] = -np.inner(self.B_diag[i], APsi2).real
            
        # solve for beta
        beta = np.linalg.lstsq(GramB, gamma, rcond=None)[0]
        
        # define evolution operators (also needed in actual evolution step)
        #beta0 = beta[-1] = -expval(H)
        G = self.A_trans + np.diag(sum([beta[i]*self.B_diag[i] for i in range(self.num_constraints)]))
        #detGram = np.linalg.det(GramB)
        #H = G - beta0*A.basis.id()
        
        return G, beta # return generator and beta

    def get_b(self, Psi: np.ndarray):
        # get expectation value for all B operators (diagonal)
        Psi2 = np.multiply(Psi, Psi.conj()) # element-wise |Psi|^2
        return [np.inner(self.B_diag[i], Psi2) for i in range(self.num_constraints)]
    
    def A_expval(self, Psi: np.ndarray):
        # expectation value of A matrix (in transformed basis)
        return np.inner(Psi.conj(), self.A_trans @ Psi).real
    
    def run(self, b: list, random_jump_method = 'phase', random_initial_method = None):
        global flip_index
        # inf expval(A) under constraints expval(B) = b where B0 = id and b0 = 1 is added
        # by performing imaginary-time evolution with potential that guarantees constraint 
        # random_jump_method can be 'phase', 'sign' and None
    
        print("co.run with b = " + str(b))
    
        # test arguments    
        if not isinstance(b, list) and not isinstance(b, np.ndarray):
            raise TypeError("'b' must be a list.")
        if len(b) != self.num_constraints-1:
            raise ValueError("Length of 'b' must fit to the number of constraints.")
    
        # add normalization constraint to b
        b = np.append(np.array(b), 1) # also make ndarray
    
        # find initial state
        # solve for |c_k|^2 with Lambda matrix = rows are diagonals of the B_i eigenvalues with linear program to get coeff in [0,1]
        # sqrt for coefficients
        linprog_res = scipy.optimize.linprog(c=np.ones(self.dim), A_eq=np.array(self.B_diag), b_eq=b, bounds=(0, 1))
        if linprog_res['status']==2: # infeasible to solve
            raise ValueError('The given constraints are not N-representable.')
        elif linprog_res['status']>0:
            raise ValueError('scipy.optimize.linprog error code {:d}'.format(linprog_res['status']))
        Psi = np.sqrt(linprog_res['x']) # coeffcients in basis that makes all B diagonal, this basis will be further used
        if random_initial_method is not None: Psi = random_phase(Psi, random_initial_method)
        
        # test constraints
        if np.linalg.norm(self.get_b(Psi) - b) > self.CONSTRAINT_ACCURACY:
            raise ValueError('Constraints not fulfilled for initial state. Maybe the given constraints are not N-representable.')

        # iterate time
        A_expval = float('NaN') # works in plt.imshow()
        A_record = []
        count_random_jump = 0
        main_converged = False
        
        for iter_main in range(self.MAX_ITER_MAIN):
            #print("iter = {}".format(iter_main))
            
            # implicit method for getting generator at delta_tau/2
            sc_converged = False
            Psi_sc = Psi
                
            for iter_sc in range(self.MAX_ITER_SC): # every trial has multiple iterations
                Psi_sc_prev = Psi_sc
                
                # get generator and its beta
                G,beta = self.solve_beta(Psi_sc)
                
                # step from original Psi with half time-step
                Psi_sc = timestep(G, Psi, self.DELTA_TAU/2, method='Euler') # no need for Crank-Nicolson here

                # test convergence
                if iter_sc>=1 and np.linalg.norm(Psi_sc - Psi_sc_prev) < self.IMPLICIT_CONVERGENCE_ACCURACY:
                    #print("Implicit half-step method converged after %d steps." % iter_sc)
                    sc_converged = True
                    print(iter_sc)
                    break
                
            if not sc_converged:
                print("Implicit half-step method not converged. Increase maximum sc iterations or decrease time-step size.")
                break
            
            # do full time step with this converged operator
            Psi = timestep(G, Psi, self.DELTA_TAU, method='Crank-Nicolson')
            
            # test constraints
            if np.linalg.norm(self.get_b(Psi) - b) > self.CONSTRAINT_ACCURACY:
                print('Constraints not fulfilled any more for propagated state. Try to decrease time-step size.')
                break
            
            # get new <A> expectation value
            A_prev = A_expval
            A_expval = self.A_expval(Psi)
            A_record.append(A_expval)
            
            # test expval(G) = 0
            if abs(A_expval + np.dot(beta,b)) > self.ENERGY_CONSTRAINT_ACCURACY:
                print('Expval(G) = 0 constraint is not fulfilled for propagated state. Try to decrease time-step size.')
                break
            
            # test convergence in energy
            if iter_main > self.MIN_ITER_MAIN-1 and abs(A_expval - A_prev) < self.ENERGY_CONVERGENCE_ACCURACY:
                print("Converged after {:d} steps.".format(iter_main), end="")
                if random_jump_method is not None: print(" ({:d} random {} jumps)".format(count_random_jump, random_jump_method), end="")
                print("") # line break
                main_converged = True
                break
            
            # do random phase shift?
            if random_jump_method is not None and self.RANDOM_PHASE_CHANCE > 0 and random.randint(1, self.RANDOM_PHASE_CHANCE)==1:
                Psi_rand = random_phase(Psi, random_jump_method)
                A_rand = self.A_expval(Psi_rand)
                if A_rand < A_expval: # keep this one
                    Psi = Psi_rand
                    count_random_jump += 1
        
        if not main_converged:
            print("Imaginary-time evolution method not converged. Increase maximum iterations.")
            
        # last beta component is beta0
        return {'min': A_expval,
                'optimizer': Psi,
                'beta': beta,
                'count_random_jump': count_random_jump,
                'A_record': A_record} 