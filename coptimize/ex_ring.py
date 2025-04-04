'''
Constrained Search in Imaginary Time
ring lattice with multiple particles example
calculate density for a given potential and invert back to potential
also run for comparison with Lieb optimization method (legendre_transform)
'''

from qmodel import LatticeBasis, Operator, OperatorList
from coptimize import ConstraintOptimization
from qmodel.dft import EnergyFunctional
from timeit import default_timer
from math import pi
import numpy as np
import matplotlib.pyplot as plt 
import pickle # io
import time

savepath = 'ex_ring/' # for pickle files
m_range = range(10, 41, 2) #range(6, 30, 1)
N = 2 # number of particles fixed, < m, N=1 is solved in zero iterations with initial value

runtime = []
runtime_comp = []
accuracy = []
accuracy_comp = []
dim = []

for m in m_range: # number of vertices
    print("")
    print("run for m={:d}".format(m))

    basis_onepart = LatticeBasis(m)
    basis = basis_onepart.wedge(N)
    
    ## ring
    #next_site = lambda qn: dict(qn, i=qn['i']%m+1)
    #H0 = -basis_onepart.sum(lambda qn: basis.hop(next_site(qn), qn).add_adj())
    
    ## linear chain
    H0 = Operator(basis)
    for i in range(1, m):
        H0 += basis.hop({'i': i}, {'i': i+1}).add_adj()
        
    B = OperatorList(basis, [ basis.hop({'i': i}) for i in range(1,m) ] ) # only up to m-1 to keep linearly independent together with B_0 = id
    
    co = ConstraintOptimization(H0, B)
    co.MAX_ITER_MAIN = 100000 # needs sometimes more
    co.ENERGY_CONVERGENCE_ACCURACY = 1e-8 # high accuracy
    dim.append(co.dim)
    print("Hilbert space size: {:d}".format(co.dim))
    
    x = np.linspace(0, 2*pi, m+1) # periodic b.c., last point in sin is first again (else doubled)
    # sin potential, not too strong else density gets too low
    #v = np.sin(x) / 10
    v = np.zeros(m+1)
    v = v[1:] # last point must be 0 for gauge fixing, anyway with sin
    H = H0 + basis_onepart.graph_pot(v).extend(basis)
    sol = H.eig(hermitian = True)
    print("gs multiplicity = {:d}".format(sol['degeneracy'][0]))
    H0_expval = H0.expval(sol['eigenvectors'][0], transform_real=True)
    print("spectral gap = {:f}".format(sol['eigenvalues'][1]-sol['eigenvalues'][0]))
    rho = sol['eigenvectors'][0].trace(basis_onepart)
    
    # run inverse problem
    start = default_timer()
    res = co.run(rho[0:-1], random_jump_method = 'phase') # remove last entry (only m-1 lattice points)
    runtime.append(default_timer()-start)
    
    vres = res['beta'] # includes beta0 as last entry
    func = res['min']
    
    accuracy.append(np.linalg.norm(vres[:-1] - v[:-1]))
    print("accuracy of potential: {:.4f}".format(accuracy[-1]))
    print("accuracy of functional: {:.6f}".format(abs(func - H0_expval)))
    
    # run comparison
    print("run comparison: Lieb optimization implemented in dft.legendre_transform()")
    E = EnergyFunctional(H0, B)
    
    start = default_timer()
    res_comp = E.legendre_transform(dens = rho[:-1], verbose = True)
    runtime_comp.append(default_timer()-start)
    
    vres_comp = res_comp['pot']
    func_comp = res_comp['F']
    
    accuracy_comp.append(np.linalg.norm(vres_comp - v[:-1]))
    print("accuracy of potential: {:.4f}".format(accuracy_comp[-1]))
    print("accuracy of functional: {:.6f}".format(abs(func_comp - H0_expval)))

    if m == m_range[-1]: # get gs density as test at last step from both potentials
        rho2 = E.solve(vres[:-1])['gs_vector'].trace(basis_onepart)
        rho2_comp = E.solve(vres_comp)['gs_vector'].trace(basis_onepart)

# save to file
with open(savepath + 'ring-comparison-' + time.strftime("%Y%m%d-%H%M%S") + '.pkl', 'wb') as f:
    pickle.dump([v, vres, vres_comp, rho, rho2, rho2_comp, m_range, dim, runtime, runtime_comp, accuracy, accuracy_comp], f)
        
# plot last result
plt.plot(v, 'b')
vres[-1] = 0 # set beta_0 to zero (last entry)
plt.plot(vres, 'bx')
plt.plot([*vres_comp, 0], 'b^') # add a zero at end
plt.plot(rho, 'r')
plt.plot(rho2, 'rx')
plt.plot(rho2_comp, 'r^')

# comparison plot
fig, (ax1,ax2) = plt.subplots(2, 1)
ax1.set_xticks(m_range, [])
ax1.set_ylabel('runtime [sec]')
ax1.plot(m_range, runtime, '-x', label="imaginary-time evolution")
ax1.plot(m_range, runtime_comp, '-^', label="BFGS comparison")
ax1.legend()
ax2.set_xticks(m_range, ["$m={:d}, L={:d}$".format(*label) for label in zip(m_range,dim)], rotation=45)
ax2.set_ylabel(r'$\Delta$ potential')
ax2.plot(m_range, accuracy, '-x')
ax2.plot(m_range, accuracy_comp, '-^')

plt.show()