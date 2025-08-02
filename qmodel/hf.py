## library for Hartree-Fock
## written for LMU iDMFT project

import itertools
import rdm
import numpy as np
import scipy
import matplotlib.pyplot as plt
from qmodel import Basis, Operator, LatticeBasis, SpinBasis, Vector
from math import sqrt, log
from scipy.optimize import curve_fit

def mf_operator(W: Operator, orbitals: list, occupation: list = None) -> Operator:
    # W needs to be a 2-particle operator on a full tensor basis (NOT (anti)symmetric)
    # (only apply to 2-body part of Hamiltonian)
    # orbitals (list of normalized Vectors) are the orbitals used for the HF operator
    # orbitals must be defined on the same 1-particle basis as W
    # occupation is the occupation number list, if not provided then all(!) orbitals are used equally (aufbau)
    # negative occupation numbers or such that they do add up to non-integer particle number are generally allowed 
    # create and return the mean-field operator
    
    if W.basis.part_num != 2:
        raise ValueError('The provided operator must be a 2-particle operator, i.e., it must be defined on a 2-particle basis.')
    elif W.basis.symmetry != 'x':
        raise ValueError('The provided operator must be defined on a 2-particle tensor basis (no symmetry restriction).')
    elif W.basis.part_basis != orbitals[0].basis:
        raise ValueError('The 1-particle basis of the provided operator and the orbitals must agree.')

    if occupation is None:
        occupation = [1]*len(orbitals) # aufbau
    elif len(orbitals) != len(occupation):
        raise ValueError('The orbital list and the occupation list must be of the same length.')
    
    # create operator as
    # W^MF_{ij} = sum_k n_k <e_i x \chi_k, W (e_j x \chi_k - \chi_k x e_j)>
    # the operation 'x' is the usual tensor product
    b1 = W.basis.part_basis
    MF = Operator(b1)
    
    for (i,j) in itertools.product(range(b1.dim), repeat=2):
        for k in range(len(orbitals)):
            if occupation[k] != 0: # (could also be ignored if very small)
                vec_ik = b1[i].tensor(orbitals[k], W.basis) # on 2-particle basis
                vec_jk = b1[j].tensor(orbitals[k], W.basis) - orbitals[k].tensor(b1[j], W.basis) # this one is antisymmetrized 
                MF.matrix[i,j] += occupation[k] * vec_ik.inner(W * vec_jk) / 2
                
    return MF

# default parameters
M = 4
N = 4
t = 1
v = [1,2,0,0]
U = 1

# init
b_lattice = LatticeBasis(M)
b_spin = SpinBasis()
b_onepart = b_lattice.tensor(b_spin) # one-particle space
next_site = lambda qn: dict(qn, i=qn['i']%M+1)

# Hamiltonian parts abstraction on basis
def ham_onepart(b: Basis, t = 1, v = None):
    # one-body part
    if v is None: v = [0]*b.part_basis.dim
    return -t*b_lattice.sum(lambda qn: b.hop(next_site(qn), qn).add_adj()) \
        + b_lattice.sum(lambda qn: v[qn['i']-1]*b.hop(qn)) \
        + b.hop({'i': 1, 's': 1}) # break up/down symmetry

def ham_W(b: Basis, U = 1):
    # two-body interaction part
    return U*b_lattice.sum(lambda qni: b.hop(dict(qni, s=1)) * b.hop(dict(qni, s=-1)))

def rdm_from_orbitals(orbitals: list, occupation: list):
    return sum([ occupation[i]*orbitals[i].proj() for i in range(len(orbitals)) ]).matrix

# Define the Fermi-Dirac function model with parameters
# also work with np.arrays
def fermi_dirac(eps, mu, kappa):
    return 1 / (np.exp((eps - mu) / kappa) + 1)

def inv_fermi_dirac(occupation, mu, kappa):
    return mu + kappa*np.log(np.reciprocal(occupation)-1)

def fit_fermi_dirac(eps_data, n_data):
    # Initial guess for parameters: mu, kappa
    initial_guess = [np.median(eps_data), 0.1]
    # Fit the model to your data
    params, _ = curve_fit(fermi_dirac, eps_data, n_data, p0=initial_guess)
    return params # mu_fit, kappa_fit

def solve_mu(eps, kappa, N): # eps is list
    # define equation whose zero gives correct mu
    eps = np.array(eps)
    def num_eqn(mu):
        N_eval = np.sum(fermi_dirac(eps, mu, kappa))
        return N_eval - N
    # choose search interval
    mu_min, mu_max = np.min(eps) - 10*kappa, np.max(eps) + 10*kappa
    # Numerically solve
    return scipy.optimize.brentq(num_eqn, mu_min, mu_max)

def test_natural_dist():
    b_full = b_onepart.wedge(N)
    H_full = ham_onepart(b_full, t, v) + ham_W(b_full, U)
    sol_full = H_full.eig(hermitian = True)
    Psi = sol_full['eigenvectors'][0]
    natural = rdm.natural(Psi)
    
    # HF operator from natural orbitals and occupations
    h1 = ham_onepart(b_onepart, t, v)
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    hf = h1 + mf_operator(W, natural['orbitals'], natural['occupation'])
    eps = [hf.expval(natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim)] # not sorted
    
    # sort for plotting and comparison
    dist = sorted(zip(eps, natural['occupation'], natural['orbitals'])) # sort by energy
    eps_data,n_data,orbs = zip(*dist)
    
    mu_fit, kappa_fit = fit_fermi_dirac(eps_data, n_data)
    print(f"Fitted mu = {mu_fit:.3f}, kappa = {kappa_fit:.3f}")
    mu_calc = solve_mu(eps, kappa_fit, N)
    print(f"Compare to calculated mu = {mu_calc:.3f}")
    
    # Generate smooth curve for plotting fitted function
    eps_fit = np.linspace(np.min(eps_data), np.max(eps_data), 200)
    n_fit = fermi_dirac(eps_fit, mu_fit, kappa_fit)
    
    # Plot data and fitted curve
    plt.plot(eps_fit, n_fit, color='red', label='Fermi-Dirac fit')
    plt.plot(eps_data, n_data, 'o-', label='data')
    
    # return 1RDM of full solution as reference
    return rdm.rdm1(Psi), rdm.entropy_ph(natural['occupation']), kappa_fit
    
full_rdm, full_entropy_ph, kappa_fit = test_natural_dist()

def test_hf():
    # test iteration to solve for gs
    # get h1 orbitals
    h1 = ham_onepart(b_onepart, t, v)
    sol = h1.eig(hermitian = True)
    orbs = sol['eigenvectors'] # for init
    
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    # HF iteration
    print('HF iteration')
    for _ in range(10):
        hf = h1 + mf_operator(W, orbs[:N]) # only occupied orbitals
        sol_hf = hf.eig(hermitian = True)
        orbs = sol_hf['eigenvectors']
    
        # HF energy
        print(sum(sol_hf['eigenvalues'][:N]))
    
    print(sol_hf['eigenvalues'])
    
    # distance to full_occupation,full_orbs in terms of 1RDM
    hf_occupation = np.zeros(b_onepart.dim)
    hf_occupation[:N] = 1
    hf_rdm = rdm_from_orbitals(orbs, hf_occupation)
    d = np.linalg.norm(hf_rdm - full_rdm)
    print(f"distance to NO: {d:.3f}")
    
    # true gs energy
    b_full = b_onepart.wedge(N)
    H_full = ham_onepart(b_full, t, v) + ham_W(b_full, U)
    sol_full = H_full.eig(hermitian = True)
    E = sol_full['eigenvalues'][0]
    print('true gs energy')
    print(E)
    Psi = sol_full['eigenvectors'][0]
    natural = rdm.natural(Psi)
    
    # compare different E_com energies to test HF operator
    # E_cum from difference to eps_HF
    hf = h1 + mf_operator(W, natural['orbitals'], natural['occupation'])
    eps_HF = sum([ natural['occupation'][i] * hf.expval(natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim) ])
    print('E_cum from difference to eps_HF (faster)')
    print(E-eps_HF)
    
    # rebuild interaction as matrix on one-particle basis like pair matrix
    W2 = np.zeros((b_onepart.dim, b_onepart.dim), dtype=np.complex128)
    for index1,e1 in enumerate(b_onepart.el):
        for index2,e2 in enumerate(b_onepart.el):
            if (e1['i']==e2['i'] and e1['s']!=e2['s']):
                W2[index1,index2] = U/2
    print('E_cum from cumulant (direct)')
    print(np.trace(np.dot(rdm.cum(Psi), W2)).real)
    
#test_hf()

def test_collins():
    # for correlated system, test distribution of natural occupation numbers and get kappa for different t
    # seems to be rather linear in t for larger t !
    # get full solution
    b_full = b_onepart.wedge(N)
    
    t_array = np.linspace(0.01, 1, 20)
    kappa_array = []
    
    for t in t_array:
        H_full = ham_onepart(b_full, t, v) + ham_W(b_full, U)
        sol_full = H_full.eig(hermitian = True)
        Psi = sol_full['eigenvectors'][0]
        natural = rdm.natural(Psi)
        
        # HF operator from natural orbitals and occupations
        h1 = ham_onepart(b_onepart, t, v)
        W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
        hf = h1 + mf_operator(W, natural['orbitals'], natural['occupation'])
        eps = [hf.expval(natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim)]
        
        # does not need to be sorted here
        _, kappa_fit = fit_fermi_dirac(eps, natural['occupation'])
        kappa_array.append(kappa_fit)
    
    # Plot data
    fig = plt.figure()
    plt.plot(t_array, kappa_array)
    plt.xlabel('$t$')
    plt.ylabel(r'$\kappa$')

#test_collins()

def test_sc_thermal_hf():
    # test self-consistent thermal HF
    # kappa is determined sc in double cycle
    # seems to converge to different kappa values, but convergence stabilized?
    # actually probably leads to premature convergence and gets stuck at kappa/orbitals
    
    # get h1 orbitals
    h1 = ham_onepart(b_onepart, t, v)
    sol = h1.eig(hermitian = True)
    orbs = sol['eigenvectors'] # for init, whole space
    eps = sol['eigenvalues']
    
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    # HF iteration
    print('sc thermal HF iteration')
    
    kappa = 0.1 # init
    
    for _ in range(10): # double cycles
        # (a) cycle: from energies get mu and occupation
        mu = solve_mu(eps, kappa, N)
        occupation = fermi_dirac(eps, mu, kappa)
        # construct HF operator from that and solve for new orbitals and energies
        hf = h1 + mf_operator(W, orbs, occupation)
        sol_hf = hf.eig(hermitian = True)
        orbs = sol_hf['eigenvectors']
        eps = sol_hf['eigenvalues']
        # (b) cycle: from the new orbs get new HF operator with previous occupation
        hf = h1 + mf_operator(W, orbs, occupation)
        sol_hf = hf.eig(hermitian = True)
        orbs = sol_hf['eigenvectors']
        eps = sol_hf['eigenvalues']
        # use new energies to update kappa
        _, kappa = fit_fermi_dirac(eps, occupation)
        # test N
        print("test N = {}".format(np.sum(fermi_dirac(eps, mu, kappa))))
        # summary
        print(f"mu = {mu:.3f}, kappa = {kappa:.3f}")
        print("thermal HF energy = {:.6f}, pure HF energy = {:.6f}".format(np.inner(eps, occupation), sum(eps[:N])))

    # distance to full_occupation,full_orbs ... is better here !
    hf_rdm = rdm_from_orbitals(orbs, occupation)
    d = np.linalg.norm(hf_rdm - full_rdm)
    print(f"distance to NO: {d:.3f}")

#test_sc_thermal_hf()

def test_thermal_hf_fixed_kappa():
    # test thermal HF with fixed kappa
    # gives same energy but better orbitals ??
    
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    # try different kappa
    kappa_array = np.linspace(0.01, kappa_fit*2, 50)
    d_array = []
    S_array = []
    for kappa in kappa_array:
        # init: get h1 orbitals
        h1 = ham_onepart(b_onepart, t, v)
        sol = h1.eig(hermitian = True)
        orbs = sol['eigenvectors'] # for init, whole space
        eps = sol['eigenvalues']
        
        # HF iteration
        print(f'thermal HF iteration kappa = {kappa:.3f}')
        
        for _ in range(10):
            # from energies get mu and occupation
            mu = solve_mu(eps, kappa, N)
            occupation = fermi_dirac(eps, mu, kappa)
            # construct HF operator from that and solve for new orbitals and energies
            hf = h1 + mf_operator(W, orbs, occupation)
            sol_hf = hf.eig(hermitian = True)
            orbs = sol_hf['eigenvectors']
            eps = sol_hf['eigenvalues']
            print("thermal HF energy = {:.6f}, pure HF energy = {:.6f}".format(np.inner(eps, occupation), sum(eps[:N])))
            
        # distance to full_occupation,full_orbs
        hf_rdm = rdm_from_orbitals(orbs, occupation)
        d = np.linalg.norm(hf_rdm - full_rdm)
        d_array.append(d)
        print(f"distance to NO: {d:.3f}")
        S_array.append(abs(full_entropy_ph - rdm.entropy_ph(occupation)))
    
    fig = plt.figure()
    plt.plot(kappa_array, d_array, 'b-', label='NO distance')
    plt.plot(kappa_array, S_array, 'r-', label='ph entropy distance')
    
    # mark min
    d_min_index = d_array.index(min(d_array))
    plt.scatter([kappa_array[d_min_index]], [d_array[d_min_index]], marker='x')

    # mark kappa_fit
    plt.axvline(x=kappa_fit, color='k', linestyle='--', label=r'$\kappa$ fit')

    plt.xlabel(r'$\kappa$')
    plt.legend()

test_thermal_hf_fixed_kappa()

def test_thermal_hf_kappa_limit():
    # test thermal HF with kappa -> 0 limit
    # gives same energy but better orbitals
    
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    # HF iteration
    print('thermal HF iteration kappa -> 0')
    
    kappa_array = np.linspace(1,0.01,20)
    for kappa in kappa_array:
        # init: get h1 orbitals
        h1 = ham_onepart(b_onepart, t, v)
        sol = h1.eig(hermitian = True)
        orbs = sol['eigenvectors'] # for init, whole space
        eps = sol['eigenvalues']
        
        # from energies get mu and occupation
        mu = solve_mu(eps, kappa, N)
        occupation = fermi_dirac(eps, mu, kappa)
        # construct HF operator from that and solve for new orbitals and energies
        hf = h1 + mf_operator(W, orbs, occupation)
        sol_hf = hf.eig(hermitian = True)
        orbs = sol_hf['eigenvectors']
        eps = sol_hf['eigenvalues']
        print("thermal HF energy = {:.6f}, pure HF energy = {:.6f}".format(np.inner(eps, occupation), sum(eps[:N])))
    
    # distance to full_occupation,full_orbs
    hf_rdm = rdm_from_orbitals(orbs, occupation)
    d = np.linalg.norm(hf_rdm - full_rdm)
    print(f"distance to NO: {d:.3f}")

#test_thermal_hf_kappa_limit()

plt.show()