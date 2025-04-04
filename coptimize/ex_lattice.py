'''
Constrained Search in Imaginary Time
small lattice or random graph with 2 particles example
calculate density for a given potential and invert back to potential
show convergence plot: <A> over iteration steps --> convergence plots for paper 
'''

from qmodel import LatticeBasis, OperatorList, timer
from coptimize import ConstraintOptimization
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 

N = 2 # number of particles, <= m

# butterfly graph (m=5)
#m = 5 # number of nodes
#edges = {(1, 2), (2, 3), (1, 3), (1, 4), (4, 5), (1, 5)}
# complete graph (maximal symmetry, m=5)
#edges = {(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)}
# square graph
m = 4
edges = {(1, 2), (2, 3), (3, 4), (4, 1)}
# random graph
#nxG = nx.gnp_random_graph(m, 0.5, 123)
#edges = set()
#for e in nxG.edges:
#    edges.add((e[0]+1,e[1]+1)) # nx edges start numbering at 0, not 1 as required here

basis_onepart = LatticeBasis(m)
basis = basis_onepart.wedge(N)

H0 = -basis_onepart.graph_laplacian(edges, include_degree=False).extend(basis)
B = OperatorList(basis, [ basis.hop({'i': i}) for i in range(1,m) ] ) # only up to m-1 to keep linearly independent together with B_0 = id

co = ConstraintOptimization(H0, B)
print("Hilbert space size: {:d}".format(co.dim))

@timer
def run_linear():
    # equidensity (central density)
    equidens = np.ones(m) * N/m
    # define one directions in dens domain (vectors that sum to 0)
    dx = np.zeros(m)
    dx[0] = N/m
    dx[1] = -N/m
    h = 0.99 # how far to move in each direction, + and -
    resol = 21 # resolution
    F = np.zeros(resol)
    linsp = np.linspace(-h,h,resol)
    num_steps = resol
    step = 1
    for ix in range(resol):
        print("step {:d} / {:d}".format(step, num_steps))
        x = linsp[ix]
        rho = equidens + x*dx
        rho = rho[0:-1] # remove last item, fixed by normalization
        # run
        res = co.run(rho, random_jump_method='phase')
        F[ix] = res['min']
        step += 1
    
    plt.plot(F)

#run_linear()

@timer
def single_inverse_runs(runs=5, random_jump_method='phase', random_initial_method=None):
    # forward problem -> solving for gs density     
    # run inverse problem
    
    co.MIN_ITER_MAIN = 5000 # all plots have same x-axis
    
    v = np.zeros(m)
    
    # two different examples in paper: with and without degeneracy
    degen = True
    
    if degen:
        # choose potential with certain symmetry -> density in degeneracy region
        # example from arXiv:2106.15370, Eq. (46) with s=t=1/2
        v[0] = 1
        v[2] = -1
    else:
        # example without degeneracy
        v[0] = 1
        v[1] = 1
    
    H = H0 + basis_onepart.graph_pot(v).extend(basis)
    sol = H.eig(hermitian = True)
    print("ground state multiplicity: {}".format(sol['degeneracy'][0]))
    H0_expval = H0.expval(sol['eigenvectors'][0], transform_real=True)
    
    if degen:
        # move into degeneracy region by combining two ground states
        rho1 = B.expval(sol['eigenvectors'][0], transform_real=True)
        rho2 = B.expval(sol['eigenvectors'][1], transform_real=True)
        rho = (rho1 + rho2)/2 # for paper
        #rho = rho1 # has qualitatively other behavior
    else:
        rho = B.expval(sol['eigenvectors'][0], transform_real=True)
    
    _, ax = plt.subplots() # new plot
    ax.set_title("square graph, random_initial_method={}, random_jump_method={}".format(random_initial_method, random_jump_method))
    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$\langle\hat A\rangle_\Psi$")
    ax.axhline(y=H0_expval, color = 'k', linewidth=1, linestyle = 'dashed') # gs energy
    ax.text(x=co.MIN_ITER_MAIN*1.065, y=H0_expval-.03, s=r"$E_0$")
    
    if degen:
        # for inset when random_jump_method='phase' w/ degeneracy
        xlim = (500, 1500)
        ylim = (-1.65, -1.2)
    else:
        # for inset when random_jump_method='phase' w/o degeneracy
        xlim = (0, 500)
        ylim = (-1, 0)
    
    if random_jump_method=='phase' and random_initial_method is None:
        # make inset to show jumps
        axins = ax.inset_axes(
            [.5, .2, .48, .78],
            xlim=xlim, ylim=ylim,
            xticklabels=[], yticklabels=[])
        inset_indicator = ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1)
        
        if degen:
            # indicator lines for example w/ degeneracy
            inset_indicator.connectors[0].set_visible(False)
            inset_indicator.connectors[1].set_visible(True)
            inset_indicator.connectors[2].set_visible(True)
            inset_indicator.connectors[3].set_visible(False)
        else:
            # indicator lines for example w/o degeneracy
            inset_indicator.connectors[0].set_visible(True)
            inset_indicator.connectors[1].set_visible(True)
            inset_indicator.connectors[2].set_visible(False)
            inset_indicator.connectors[3].set_visible(False)
    
    for n in range(runs): # multiple runs
        print("run {:d}".format(n+1))

        res = co.run(rho, random_jump_method=random_jump_method, random_initial_method=random_initial_method)
        vres = res['beta']
        
        print("potential result from inversion = ", end="")
        print(vres[:-1])
        print("accuracy of potential: {:.4f}".format(np.linalg.norm(vres[:-1] - v[:-1])))
        print("accuracy of functional: {:.4f}".format(abs(res['min'] - H0_expval)))

        ax.plot(res['A_record'][0:co.MIN_ITER_MAIN])
        if random_jump_method=='phase' and random_initial_method is None:
            axins.plot(res['A_record'][0:co.MIN_ITER_MAIN]) # plot into inset


# multiple runs with different parameters
# this creates the convergence plots for the paper

#single_inverse_runs(runs=1, random_jump_method=None, random_initial_method=None)
#single_inverse_runs(random_jump_method=None, random_initial_method='sign')
#single_inverse_runs(random_jump_method=None, random_initial_method='phase')
#single_inverse_runs(random_jump_method='sign', random_initial_method=None) # not that interesting
single_inverse_runs(random_jump_method='phase', random_initial_method=None)

plt.show()