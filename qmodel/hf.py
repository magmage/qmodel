## library for Hartree-Fock
## written for LMU iDMFT project

# move examples to "examples" dir

import itertools
import rdm
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle
from qmodel import Basis, Operator, LatticeBasis, SpinBasis, Vector
from math import sqrt, log, exp
from scipy.optimize import curve_fit

def mf_operator(W: Operator, orbitals: list, occupation: list = None) -> Operator:
    # W needs to be a 2-particle operator on a full tensor basis (NOT (anti)symmetric)
    # (only apply to 2-body part of Hamiltonian)
    # orbitals (list of normalized Vectors) are the orbitals used for the HF operator
    # orbitals must be defined on the same 1-particle basis as W
    # occupation is the occupation number list, if not provided then all(!) orbitals are used equally (aufbau)
    # negative occupation numbers or such that they do add up to non-integer particle number are generally allowed 
    # create and return the mean-field operator (*without* 1/2 factor)
    
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
                MF.matrix[i,j] += occupation[k] * vec_ik.inner(W * vec_jk) # no factor 1/2 in MF (single-particle) operator
                
    return MF

# from here on it is specialized to Hubbard rings
# default parameters
M = 4
N = 4
t = 1 # need 1/2 if M=2 else double counted
v = [0,0,0,0,0]
U = 1

# init
b_lattice = LatticeBasis(M)
b_spin = SpinBasis()
b_onepart = b_lattice.tensor(b_spin) # one-particle space
b_full = b_onepart.wedge(N)
next_site = lambda qn: dict(qn, i=qn['i']%M+1)

# Hamiltonian parts abstraction on basis
def ham_onepart(b: Basis, t = 1, v = None):
    # one-body part
    if v is None: v = [0]*b.part_basis.dim
    return -t*b_lattice.sum(lambda qn: b.hop(next_site(qn), qn).add_adj()) \
        + b_lattice.sum(lambda qn: v[qn['i']-1]*b.hop(qn)) \
        + 0.001*b.hop({'s': 1}) # slightly break up/down symmetry

def ham_W(b: Basis, U = 1):
    # two-body interaction part
    return U*b_lattice.sum(lambda qni: b.hop(dict(qni, s=1)) * b.hop(dict(qni, s=-1)))

def rdm_from_orbitals(orbitals: list, occupation: list):
    return sum([ occupation[i]*orbitals[i].proj() for i in range(len(orbitals)) ]).matrix

def hf(h1, W, kappa = 0): # also thermal HF
    # init: get h1 orbitals
    sol = h1.eig(hermitian = True)
    orbs = sol['eigenvectors'] # for init, whole space
    eps = sol['eigenvalues']

    if kappa < 0:
        raise ValueError('kappa (temperature) cannot be negative')
    elif kappa == 0:
        occupation = np.zeros(h1.basis.dim)
        occupation[:N] = 1        
    
    for _ in range(10):
        if kappa > 0: # positive temperature: from current eps get mu and occupation
            mu = solve_mu(eps, kappa, N)
            occupation = fermi_dirac(eps, mu, kappa)
        mf = mf_operator(W, orbs, occupation)
        hf = h1 + mf
        sol_hf = hf.eig(hermitian = True)
        orbs = sol_hf['eigenvectors']
        eps = sol['eigenvalues']
    
    # to get the HF energy avoid double counting from eps_i
    E_HF = sum([ occupation[i] * (h1 + 1/2*mf).expval(orbs[i], transform_real = True) for i in range(h1.basis.dim) ])
    return E_HF, occupation, eps, orbs # HF energy, eps, orbs -> whole space
     
# Define the Fermi-Dirac function model with parameters
# also work with np.arrays
def fermi_dirac(eps, mu, kappa):
    if kappa < 0:
        raise ValueError('kappa (temperature) cannot be negative')
    eps_scaled_clipped = np.clip((np.array(eps) - mu) / kappa, -80, 80)
    return 1 / (np.exp(eps_scaled_clipped) + 1)

def inv_fermi_dirac(occupation, mu, kappa):
    return mu + kappa*np.log(np.reciprocal(occupation)-1)

def fit_fermi_dirac(eps_data, n_data):
    # Initial guess for parameters: mu, kappa
    initial_guess = [np.median(eps_data), 0.1]
    bounds = ([min(eps_data),0], [max(eps_data),np.inf])
    # Fit the model to your data
    params, _ = curve_fit(fermi_dirac, eps_data, n_data, p0=initial_guess, bounds=bounds)
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

def rdm_distance(r1, r2): # orbital set distance measured in terms of 1RDMs
    return np.linalg.norm(r1 - r2, ord=2) # operator norm is used

def gibbs_state(H, kappa):
    D = scipy.linalg.expm(-H.matrix / kappa)
    return D / np.trace(D) # normalize

def entropy_vN(D): # von Neumann entropy for a density matrix
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(D)
    # Filter out very small (zero) eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    # Compute entropy in bits (log base e)
    return -np.sum(eigenvalues * np.log(eigenvalues))

def rdm1DM(D): # from density matrix
    # return 1RDM as matrix in basis of Vector
    gamma = np.zeros((b_onepart.dim, b_onepart.dim), dtype=np.complex128)
    for index1,e1 in enumerate(b_onepart.el):
        for index2,e2 in enumerate(b_onepart.el):
            gamma[index1,index2] = np.trace(D @ b_full.hop(e2,e1).matrix) ## should use symmetry
    return gamma
 
def test_natural_dist():
    H_full = ham_onepart(b_full, t, v) + ham_W(b_full, U)
    sol_full = H_full.eig(hermitian = True)
    Psi = sol_full['eigenvectors'][0]    
    
    # print(b_full)
    # print(Psi)
    
    natural = rdm.natural(Psi)
    print('true gs energy: {:.6f}'.format(sol_full['eigenvalues'][0]))
    print('gs degeneracy: {}'.format(sol_full['degeneracy'][0]))
    
    # print(b_onepart)
    #
    # for i in range(b_onepart.dim):
    #     print('orbital n_{} = {:.6f}'.format(i+1, natural['occupation'][i]))
    #     print(natural['orbitals'][i])
    
    # HF operator from natural orbitals and occupations
    h1 = ham_onepart(b_onepart, t, v)
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    mf = mf_operator(W, natural['orbitals'], natural['occupation'])
    e_array = [(h1 + 1/2*mf).expval(natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim)]
    eps_array = [(h1 + mf).expval(natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim)] # double counting, not sorted
    
    # sort for plotting and comparison
    dist = sorted(zip(eps_array, e_array, natural['occupation'])) # sort by energy
    eps_array,e_array,n_array = zip(*dist)
    
    print('eps_i: ' + str(eps_array))
    print('e_i: ' + str(e_array))
    print('n_i: ' + str(n_array))
    eps_HF = np.inner(e_array, n_array)
    print('sum_i n_i e_i = ' + str(eps_HF))
    print('E_cum from difference to E_0 = ' + str(sol_full['eigenvalues'][0] - eps_HF))
    
    mu_fit, kappa_fit = fit_fermi_dirac(eps_array, n_array)
    print(f"Fitted mu = {mu_fit:.3f}, kappa = {kappa_fit:.3f}")
    mu_calc = solve_mu(eps_array, kappa_fit, N)
    print(f"Compare to calculated mu = {mu_calc:.3f}")
    
    # Generate smooth curve for plotting fitted function
    eps_fit = np.linspace(np.min(eps_array), np.max(eps_array), 200)
    n_fit = fermi_dirac(eps_fit, mu_fit, kappa_fit)
    
    # Plot data and fitted curve
    plt.plot(eps_fit, n_fit, color='red', label='Fermi-Dirac fit')
    plt.plot(eps_array, n_array, 'o-', label='data')
    plt.title(r'Fermi-Dirac fit: $M$={}, $N$={}, $t$={}, $U$={}, $v$={}, $\kappa$={:.3f}'.format(M,N,t,U,v[:M],kappa_fit))
    
    # return 1RDM of full solution as reference
    return Psi, sol_full['eigenvalues'][0], rdm.rdm1(Psi), rdm.entropy_ph(natural['occupation']), natural, kappa_fit
    
Psi_gs, full_gs_energy, full_rdm, full_entropy_ph, full_natural, kappa_fit = test_natural_dist() # needed for other functions as reference

def test_hf():
    # test HF iteration to solve for gs
    print('HF iteration')
    h1 = ham_onepart(b_onepart, t, v)
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    E_HF, _, _, orbs = hf(h1, W)
    print("HF energy = {:.6f}".format(E_HF))
    
    # distance to full_occupation,full_orbs in terms of 1RDM
    hf_occupation = np.zeros(b_onepart.dim)
    hf_occupation[:N] = 1
    hf_rdm = rdm_from_orbitals(orbs, hf_occupation)
    d = rdm_distance(hf_rdm, full_rdm)
    print(f"distance to NO: {d:.6f}")
    
    # test HF iteration to solve for gs
    kappa = 0.005
    print(f'thermal HF iteration kappa={kappa:.6f}')
    E_HF, hf_occupation, _, orbs = hf(h1, W, kappa)
    print("HF energy = {:.6f}".format(E_HF))
    
    # distance to full_occupation,full_orbs in terms of 1RDM
    hf_rdm = rdm_from_orbitals(orbs, hf_occupation)
    d = rdm_distance(hf_rdm, full_rdm)
    print(f"distance to NO: {d:.6f}")
    
    # compare different E_com energies to test HF operator
    # E_cum from difference to eps_HF
    h1 = ham_onepart(b_onepart, t, v)
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    Hx = h1 + 1/2*mf_operator(W, full_natural['orbitals'], full_natural['occupation']) # 1p + Hartree + exchange: needs 1/2 for E_cum
    eps_HF = sum([ full_natural['occupation'][i] * Hx.expval(full_natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim) ])
    print(eps_HF)
    print('E_cum from difference to eps_HF')
    print(full_gs_energy-eps_HF)
    
    # rebuild interaction as matrix on one-particle basis like pair matrix
    W2 = np.zeros((b_onepart.dim, b_onepart.dim), dtype=np.complex128)
    for index1,e1 in enumerate(b_onepart.el):
        for index2,e2 in enumerate(b_onepart.el):
            if (e1['i']==e2['i'] and e1['s']!=e2['s']):
                W2[index1,index2] = U/2
    print('E_cum from cumulant')
    print(np.trace(np.dot(rdm.cum(Psi_gs), W2)).real)
    
#test_hf()

def test_collins():
    # for correlated system, test distribution of natural occupation numbers and get kappa for different t
    # need more data points -> Pentamer
    # get full solution
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    t_array = np.linspace(0.01, 2, 40)
    kappa_array = []
    d_HF_array = []
    d_thHF_array = []
    d_E_array = []
    d_S_array = []
    E_cum_array = []
    S_array = []
    
    for t in t_array:
        H_full = ham_onepart(b_full, t, v) + ham_W(b_full, U)
        sol_full = H_full.eig(hermitian = True)
        Psi = sol_full['eigenvectors'][0]
        natural = rdm.natural(Psi)
        full_rdm = rdm.rdm1(Psi)
        
        # HF operator from natural orbitals and occupations
        h1 = ham_onepart(b_onepart, t, v)
        W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
        mf = mf_operator(W, natural['orbitals'], natural['occupation'])
        e_array = [(h1 + 1/2*mf).expval(natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim)]
        eps_array = [(h1 + mf).expval(natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim)]
        E_cum = sol_full['eigenvalues'][0] - np.inner(natural['occupation'], e_array)
        S = rdm.entropy_ph(natural['occupation'])
        E_cum_array.append(E_cum)
        S_array.append(S)

        # does not need to be sorted here
        _, kappa_fit = fit_fermi_dirac(eps_array, natural['occupation'])
        kappa_array.append(kappa_fit)
        
        # compare HF
        h1 = ham_onepart(b_onepart, t, v)
        _, _, _, orbs = hf(h1, W)
        hf_occupation = np.zeros(b_onepart.dim)
        hf_occupation[:N] = 1
        hf_rdm = rdm_from_orbitals(orbs, hf_occupation)
        d_HF_array.append( rdm_distance(hf_rdm, full_rdm) )
        
        # compare adapted thermal HF
        E_HF, occupation, _, orbs = hf(h1, W, kappa_fit)
        hf_rdm = rdm_from_orbitals(orbs, occupation)
        hf_S = rdm.entropy_ph(occupation)
        d_thHF_array.append( rdm_distance(hf_rdm, full_rdm) )
        d_E_array.append( E_HF - kappa_fit*hf_S - sol_full['eigenvalues'][0] ) # b of Collins conjecture
        d_S_array.append( hf_S - S )
    
    # Plot data
    plt.figure()
    plt.plot(t_array, kappa_array, 'b-', label=r'$\kappa$ fit')
    plt.plot(t_array, d_HF_array, 'g-', label='HF NO distance')
    plt.plot(t_array, d_thHF_array, 'r-x', label='thermal HF NO distance')
    plt.plot(t_array, d_E_array, 'y-', label='Helmholtz free energy of thHF - gs energy (=b)')
    plt.plot(t_array, d_S_array, 'b--', label='entropy mismatch')
    #plt.plot(t_array, S_array, 'k--', label='entropy from interaction')
    plt.xlabel('$t$')
    plt.legend()
    
    # plot Collins conjecture plot
    plt.figure()
    plt.plot(S_array, -np.array(E_cum_array))
    plt.xlabel('$S$')
    plt.ylabel(r'$-E_\mathrm{cum}$')

#test_collins()

def test_sc_thermal_hf():
    # test self-consistent thermal HF
    # kappa is determined sc in double cycle
    # actually leads to premature convergence and gets stuck at kappa/orbitals
    # also no reason why it should be different
    
    # get h1 orbitals
    h1 = ham_onepart(b_onepart, t, v)
    sol = h1.eig(hermitian = True)
    orbs = sol['eigenvectors'] # for init, whole space
    eps = sol['eigenvalues']
    
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    # HF iteration
    print('sc thermal HF iteration')
    
    kappa = 1 # init
    
    for _ in range(10): # double cycles
        # (a) cycle: from energies get mu and occupation
        mu = solve_mu(eps, kappa, N)
        occupation = fermi_dirac(eps, mu, kappa)
        # construct HF operator from that and solve for new orbitals and energies
        mf = mf_operator(W, orbs, occupation)
        hf = h1 + mf
        sol_hf = hf.eig(hermitian = True)
        orbs = sol_hf['eigenvectors']
        eps = sol_hf['eigenvalues']
        # (b) cycle: from the new orbs get new HF operator with previous occupation
        mf = mf_operator(W, orbs, occupation)
        hf = h1 + mf
        sol_hf = hf.eig(hermitian = True)
        orbs = sol_hf['eigenvectors']
        eps = sol_hf['eigenvalues']
        # use new energies to update kappa
        _, kappa = fit_fermi_dirac(eps, occupation)
        # test N
        print("test N = {}".format(np.sum(fermi_dirac(eps, mu, kappa))))
        # summary
        print(f"mu = {mu:.3f}, kappa = {kappa:.3f}")
        E_HF = sum([ occupation[i] * (h1 + 1/2*mf).expval(orbs[i], transform_real = True) for i in range(h1.basis.dim) ])
        print("thermal HF energy = {:.6f}".format(E_HF))

    # distance to full_occupation,full_orbs ... is better here !
    hf_rdm = rdm_from_orbitals(orbs, occupation)
    d = rdm_distance(hf_rdm, full_rdm)
    print(f"distance to NO: {d:.3f}")

#test_sc_thermal_hf()

def test_thermal_hf_fixed_kappa():
    # test thermal HF with fixed kappa
    # gives same energy but better orbitals ??
    h1 = ham_onepart(b_onepart, t, v)
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    # try different kappa
    kappa_array = np.linspace(0.001, kappa_fit*2, 50)
    d_array = []
    S_array = []
    for kappa in kappa_array:
        # HF iteration
        print(f'thermal HF iteration kappa = {kappa:.3f}')
        E_HF, occupation, _, orbs = hf(h1, W, kappa)
        print("thermal HF energy = {:.6f}".format(E_HF))
            
        # distance to full_occupation,full_orbs
        hf_rdm = rdm_from_orbitals(orbs, occupation)
        d = rdm_distance(hf_rdm, full_rdm)
        d_array.append(d)
        S_array.append(abs(full_entropy_ph - rdm.entropy_ph(occupation)))
    
    plt.figure()
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
    # gives same energy but better orbitals ?
    h1 = ham_onepart(b_onepart, t, v)
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF

    # HF iteration
    print('thermal HF iteration kappa -> 0')
    
    kappa_array = np.linspace(1,0.01,20)
    for kappa in kappa_array:
        E_HF, occupation, eps, orbs = hf(h1, W, kappa)
        print("thermal HF energy = {:.6f}".format(E_HF))
    
    # distance to full_occupation,full_orbs
    hf_rdm = rdm_from_orbitals(orbs, occupation)
    d = rdm_distance(hf_rdm, full_rdm)
    print(f"distance to NO: {d:.3f}")

#test_thermal_hf_kappa_limit()

def test_functionals():
    # get functionals of interacting system and for thermal HF
    # directly from E(kappa) = F(S) - kappa*S relation at solution point
    # map different kappa to S
    
    H_full = ham_onepart(b_full, t, v) + ham_W(b_full, U)
    
    h1 = ham_onepart(b_onepart, t, v)
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    kappa_array = np.linspace(0.01, 5, 200)

    S1_array = []
    F1_array = []
    S0_array = []
    F0_array = []
    
    for kappa in kappa_array:
        # 1. interacting system
        D = gibbs_state(H_full, kappa)
        E = np.trace(H_full.matrix @ D).real
        #S = entropy_vN(D) # but must use ph entropy?
        S = rdm.entropy_ph(np.linalg.eigvalsh(rdm1DM(D)))
        S1_array.append(S)
        F1_array.append(E + kappa*S)
        
        # 2. thermal HF system
        E_HF, occupation, eps, _ = hf(h1, W, kappa)
        S = rdm.entropy_ph(occupation)
        S0_array.append(S)
        F0_array.append(E + kappa*S)
        
    # difference from interpolation
    # 1. Choose common x points (e.g., union or intersection)
    S_common = np.linspace(max(S1_array[0], S0_array[0]), min(S1_array[-1], S0_array[-1]), 100)
    # 2. Interpolate each curve to the common x points, cubic spline interpolation
    F1_interp = scipy.interpolate.make_interp_spline(S1_array, F1_array)
    F0_interp = scipy.interpolate.make_interp_spline(S0_array, F0_array)
    kappa1_interp = scipy.interpolate.make_interp_spline(S1_array, kappa_array)
    kappa0_interp = scipy.interpolate.make_interp_spline(S0_array, kappa_array)    
    # 3. Subtract
    F_diff_spline = scipy.interpolate.make_interp_spline(S_common, F1_interp(S_common) - F0_interp(S_common))
    # differential
    # need spline to be able to evaluate it any point for KS
    #dF1_interp = F1_interp.derivative()
    #dF0_interp = F0_interp.derivative()
    dF_diff_spline = scipy.interpolate.make_interp_spline(S_common, kappa1_interp(S_common) - kappa0_interp(S_common)) #F_diff_spline.derivative()
    
    S_min = min(S_common)
    S_max = b_onepart.dim*log(2)

    # try and perform KS iteration for full system at kappa_full
    print('try KS iteration')
    kappa_full = 0.2 ## should also work with 0
    S_full = scipy.interpolate.interp1d(kappa_array, S1_array, kind='linear', fill_value="extrapolate")(kappa_full)
    kappa = 1 # init
    for _ in range(10):
        E_HF, occupation, eps, _ = hf(h1, W, kappa)
        S = rdm.entropy_ph(occupation)
        kappa = kappa_full - dF_diff_spline(S) # new kappa
        print(f'KS iter S={S:.3f}, kappa={kappa:.3f}')
    print(f'ph entropy full system at kappa={kappa_full:.3f}: {S_full:.3f}')
    print(f'ph entropy ref system at kappa={kappa:.3f}: {S:.3f}')
    
    plt.figure()
    plt.plot(S1_array, F1_array, 'b-', label='F full')
    plt.plot(S0_array, F0_array, 'r-', label='F ref (thHF)')
    plt.plot(S1_array, kappa1_interp(S1_array), 'b--', label='kappa full')
    plt.plot(S0_array, kappa0_interp(S0_array), 'r--', label='kappa ref (thHF)')
    #plt.plot(S1_array, dF1_interp(S1_array), 'bx', label='dF full')
    #plt.plot(S0_array, dF0_interp(S0_array), 'rx', label='dF ref (thHF)')
    plt.plot(S_common, F_diff_spline(S_common), 'g-', label='F diff')
    plt.plot(S_common, dF_diff_spline(S_common), 'g--', label='dF diff')
    
    # mark maximum entropy
    plt.axvline(x=S_max, color='k', linestyle='--', label='max entropy')
    
    plt.xlabel(r'$S$')
    plt.legend()

#test_functionals()

def test_Paul_data():
    # compare Paul data
    # Hubbard trimer U=1
    f = open('../examples/hubbard-polymer-iDMFT/Paul_Exp_Val_Fock_Op.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    
    W_full = ham_W(b_full, U)
    W = ham_W(b_onepart.ntensor(2), U) # on 2-particle tensor basis for HF
    
    V_values = [0.001, 0.5, 0.999, 1, 1.001, 1.5]
    for v1 in V_values:
        i = 0
        v = [v1, 0, 0]
        #print(data[f'NON (V={v1})'])
        #print(data[f'Fock value (V={v1})'])
        for t in data['Hopping t']:
            # calculate HF orbital energies from natural orbitals
            H_full = ham_onepart(b_full, t, v) + W_full
            sol_full = H_full.eig(hermitian = True)
            Psi = sol_full['eigenvectors'][0]
            natural = rdm.natural(Psi)
            # HF operator from natural orbitals and occupations
            h1 = ham_onepart(b_onepart, t, v)
            Hx = h1 + 1/2*mf_operator(W, natural['orbitals'], natural['occupation']) # 1p + Hartree + exchange: needs 1/2 as many-particle operator
            eps = [Hx.expval(natural['orbitals'][i], transform_real = True) for i in range(b_onepart.dim)] # not sorted
            print('NON my data')
            print(natural['occupation'])
            #print(eps)

            print('NON Paul data')
            print(data[f'NON (V={v1})'][i])
            #print(data[f'Fock value (V={v1})'][i])
            i += 1
    
#test_Paul_data()

plt.show()