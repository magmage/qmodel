'''
Constrained Search in Imaginary Time
method for calculating F(rho) functional of 4-site lattice with 2 particles (square and tetrahedron graph)
'''

from qmodel import LatticeBasis, OperatorList, timer
from coptimize import ConstraintOptimization
from multiprocessing import Pool
from functools import partial
import numpy as np
import matplotlib.pyplot as plt 
import pickle # io
import time

def plot_plane(F):
    plt.figure()
    plt.imshow(F, cmap='managua_r', interpolation='none')
    plt.colorbar()
    ax = plt.gca()
    ax.set_axis_off()

def co_single_process(rho, co, random_jump_method, random_initial_method):
    # just return functional value and discard rest
    res = co.run(rho, random_jump_method=random_jump_method, random_initial_method=random_initial_method)
    return res['min']

@timer
def run_plane(random_jump_method='phase', random_initial_method=None):
    # cut a plane horizontally through the upper part of the octahedron in the square graph example
    # h gives location of plane, h=1 top tip, h=0 is the center plane
    #resol = 101 # resolution
    resol = 101
    h = 0.0 # between 0 and 1
    eps = 0 # away from boundary (not necessary)
    linsp = np.linspace(0+eps,1-eps,resol)
    arg_list = [] # for multiprocessing map
    for iy in range(resol): # first rows
        for ix in range(resol):
            x = linsp[ix]
            y = linsp[iy]
            rho = (1-h)*(V1 + x*(V2-V1) + y*(V4-V1)) + h*Vtop
            rho = rho[0:-1] # remove last item, fixed by normalization
            arg_list.append(rho)
    
    # create single-argument callable for multiprocessing map
    f = partial(co_single_process, co=co, random_jump_method=random_jump_method, random_initial_method=random_initial_method)
    # run
    map_res = multiprocessing_pool.map(f, arg_list)
    # retrieve serialized data
    F = np.reshape(map_res, (resol,resol))
        
    # save to file
    with open(savepath + 'four-sites-F-plane-' + time.strftime("%Y%m%d-%H%M%S") + '.pkl', 'wb') as f:
        pickle.dump([F, resol, h, eps, edges, vars(co), random_jump_method, random_initial_method], f)
    
    # plot
    plot_plane(F)
    plt.title("random_initial_method={}, random_jump_method={}".format(random_initial_method, random_jump_method))


def read_plane(filename):
    # read from file
    with open(savepath + filename, 'rb') as f:
        F, resol, h, eps, edges, co_vars, random_jump_method, random_initial_method = pickle.load(f)
    print(co_vars) # for parameters
    print(edges)

    plot_plane(F)

@timer
def run_diagonal(random_jump_method='phase', random_initial_method=None):
    # cut plane diagonally
    resol = 11 # resolution
    h = 0.0
    F = np.zeros(resol)
    linsp = np.linspace(0,1,resol)
    for ix in range(resol):
        print("step {:d} / {:d}".format(ix+1, resol))
        x = linsp[ix]
        rho = (1-h)*((1-x)*V1 + x*V3) + h*Vtop # first diagonal
        #rho = (1-h)*((1-x)*V2 + x*V4) + h*Vtop # second diagonal
        print("run rho = ", end="")
        print(rho)
        rho = rho[0:-1] # remove last item, fixed by normalization
        # run
        # random_jump_method None or 'sign' does not give convex result along first diagonal
        res = co.run(rho, random_jump_method=random_jump_method, random_initial_method=random_initial_method) 
        F[ix] = res['min']
    
    plt.plot(F)

@timer
def run_single(rho):
    print("run rho = ", end="")
    print(rho)
    rho = rho[0:-1] # remove last item, fixed by normalization
    # run
    res = co.run(rho, random_jump_method='phase')
    # plot
    _, ax = plt.subplots() # new plot
    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$\langle\hat A\rangle_\Psi$")
    ax.plot(res['A_record'])


if __name__ == '__main__': # for multiprocess
    m = 4 # number of nodes
    N = 2 # number of particles, <= m
    
    savepath = 'ex_four_sites/' # for pickle files
    
    # edges for tetrahedron graph
    #edges = {(1, 2), (2, 3), (3, 4), (1, 4), (1, 3), (2, 4)}
    # edges for square graph
    edges = {(1, 2), (2, 3), (3, 4), (4, 1)}
    # edges for less symmetric graph
    #edges = {(1, 2), (2, 3), (3, 1), (1, 4)}
    
    basis_onepart = LatticeBasis(m)
    basis = basis_onepart.wedge(N)
    
    H0 = -basis_onepart.graph_laplacian(edges, include_degree=False).extend(basis)
    B = OperatorList(basis, [ basis.hop({'i': i}) for i in range(1,m) ] ) # only up to m-1 to keep linearly independent together with B_0 = id
    
    co = ConstraintOptimization(H0, B)
    co.MAX_ITER_MAIN = 500000
    co.MIN_ITER_MAIN = 5000 # in some parts of the center plane (always degeneracy) a lot of random phase jumps are necessarily to converge to the correct value
    co.DELTA_TAU = 0.00005 # the square graph else has problematic densities (like [0.77777778, 0.88888889, 0.22222222, 0.11111111]) in the shape of arcs
    
    # octahedron vertices
    Vtop = np.array([1,0,1,0])
    Vbottom = np.array([0,1,0,1])
    V1 = np.array([1,1,0,0]) # top left
    V2 = np.array([0,1,1,0]) # top right
    V3 = np.array([0,0,1,1]) # bottom right
    V4 = np.array([1,0,0,1]) # bottom left
    
    # only run_plane uses multiprocessing
    #multiprocessing_pool = Pool()
    #run_plane(random_jump_method='phase', random_initial_method=None)
    #run_plane(random_jump_method='phase', random_initial_method='phase')

    read_plane('four-sites-F-plane-20250227-211714_iniNone_jmpPhase.pkl')
    read_plane('four-sites-F-plane-20250228-022827_iniPhase_jmpPhase.pkl')
    
    # in square graph:
    # [0.96, 0.66, 0.04, 0.34] and [0.73, 0.92, 0.27, 0.08] had Expval(G) = 0 constraint not fulfilled with larger DELTA_TAU
    # rho = [0.5, 0.,  0.5, 1. ] had implicit sc iteration not converging (but on boundary anyway)
    #run_single([0.96, 0.66, 0.04, 0.34])
    
    #run_diagonal(random_jump_method='phase', random_initial_method=None)
    #run_diagonal(random_jump_method='phase', random_initial_method='phase')
    
    plt.show()