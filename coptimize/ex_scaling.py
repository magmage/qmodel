'''
Constrained Search in Imaginary Time
small lattice or random graph with 2 particles example
calculate density for a given potential and invert back to potential
show convergence plot: <A> over iteration steps --> convergence plots for paper 
'''

from qmodel import LatticeBasis, OperatorList
from coptimize import ConstraintOptimization
from timeit import default_timer
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt 
import matplotlib.cm as colormap

N = 2 # number of particles, <= m
m_range = range(4, 21) # graph sizes, actually M in draft
min_iter_main = 1000
random_jump_method = None # to have comparison don't add any random element...
random_initial_method = None 
t_array = []
iter_array = []
max_iter = 0

_, ax = plt.subplots() # new plot
ax.set_title("ring graphs, random_initial_method={}, random_jump_method={}".format(random_initial_method, random_jump_method))
ax.set_xlabel("iterations")
ax.set_ylabel(r"$\langle\hat H_0\rangle_\Psi$")
ax.axhline(y=0, color = 'k', linewidth=1, linestyle = 'dashed') # gs energy shifted to 0
colors = [colormap.coolwarm(l / len(m_range)) for l in range(len(m_range))]

for j in range(len(m_range)):
    m = m_range[j]
    print("ring graph with M={:d}".format(m))

    basis_onepart = LatticeBasis(m)
    basis = basis_onepart.wedge(N)

    # # random graph
    # nxG = nx.gnp_random_graph(m, 0.5, 123)
    # edges = set()
    # for e in nxG.edges:
    #     edges.add((e[0]+1,e[1]+1)) # nx edges start numbering at 0, not 1 as required here
    # H0 = -basis_onepart.graph_laplacian(edges, include_degree=False).extend(basis)
    
    ## ring
    next_site = lambda qn: dict(qn, i=qn['i']%m+1)
    H0 = -basis_onepart.sum(lambda qn: basis.hop(next_site(qn), qn).add_adj())
    B = OperatorList(basis, [ basis.hop({'i': i}) for i in range(1,m) ] ) # only up to m-1 to keep linearly independent together with B_0 = id
    
    co = ConstraintOptimization(H0, B)
    co.MIN_ITER_MAIN = min_iter_main # all plots have same x-axis
    print("Hilbert space size: {:d}".format(co.dim))

    # forward problem -> solving for gs density 
    v = np.zeros(m)
    v[0] = 0.1 # break symmetry
    # run inverse problem
    H = H0 + basis_onepart.graph_pot(v).extend(basis)
    sol = H.eig(hermitian = True)
    print("ground state multiplicity: {}".format(sol['degeneracy'][0]))
    H0_expval = H0.expval(sol['eigenvectors'][0], transform_real=True)
    rho = B.expval(sol['eigenvectors'][0], transform_real=True)
    
    t_start = default_timer()
    res = co.run(rho, random_jump_method=random_jump_method, random_initial_method=random_initial_method)
    t_array.append(default_timer() - t_start)
    vres = res['beta']
    
    print("potential result from inversion = ", end="")
    print(vres[:-1])
    print("accuracy of potential: {:.4f}".format(np.linalg.norm(vres[:-1] - v[:-1])))
    print("accuracy of functional: {:.4f}".format(abs(res['min'] - H0_expval)))

    ax.plot(np.array(res['A_record']) - H0_expval, color=colors[j], label=f'$M={m}$, dim={co.dim}') # shift min to 0
    
    if len(res['A_record']) > max_iter: max_iter = len(res['A_record'])
    iter_array.append(len(res['A_record']))

# plot dashed line for min
ax.text(x=max_iter*1.065, y=-.03, s=r"$E_0$")
ax.legend()

# plot runtime
_, ax = plt.subplots() # new plot
ax.plot(m_range, t_array, 'x--', color='b')
ax.set_ylabel("runtime [s]", color='b')
ax.tick_params(axis='y', labelcolor='b')
ax.set_xticks(m_range)
ax.set_xticklabels(["$M={:d}$".format(m) for m in m_range], rotation=45)
ax2 = ax.twinx()
ax2.plot(m_range, iter_array, 'x--', color='r')
ax2.set_ylabel('iterations', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.show()