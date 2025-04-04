'''
Constrained Search in Imaginary Time
cuboctahedron graph with 2 particles example
cf. arXiv:2106.15370 Sec. VI.E
show non-pure-state v-rep density !
'''

from qmodel import LatticeBasis, OperatorList, timer
from coptimize import ConstraintOptimization
from multiprocessing import Pool
from functools import partial
import numpy as np
import matplotlib.pyplot as plt 
import pickle # io
import time

def plot(F):    
    plt.imshow(F, cmap='YlGnBu', interpolation='none')
    plt.colorbar()

@timer
def run_linear():
    # equidensity (central density)
    equidens = np.array([1,1,1,1,1,1,1,1,1,1,1,1]) / 6
    # define one directions in dens domain (vectors that sum to 0)
    # not all directions work, i.e. the method does not converge correctly at those densities
    #dx = np.array([1,-1,0,0,0,0,0,0,0,0,0,0]) / 6 # works
    #dx = np.array([0,0,1,-1,0,0,0,0,0,0,0,0]) / 6 # had 1/11 wrong covergences
    dx = np.array([1,-0.4,0.2,-0.6,-0.1,-0.1,0,0,0,0,0,0]) / 6 # less symmetric
    h = 0.99 # how far to move in each direction, + and -
    resol = 11 # resolution
    linsp = np.linspace(-h,h,resol)
    arg_list = [] # for multiprocessing map
    for ix in range(resol):
        x = linsp[ix]
        rho = equidens + x*dx
        rho = rho[0:-1] # remove last item, fixed by normalization
        arg_list.append(rho)

    # create single-argument callable for multiprocessing map
    f = partial(co.run, random_jump_method='phase', random_initial_method='phase')
    # run
    multiprocessing_pool = Pool()
    map_res = multiprocessing_pool.map(f, arg_list)
    F = [res['min'] for res in map_res] # extract min

    # save to file
    with open(savepath + 'cubo-F-line-' + time.strftime("%Y%m%d-%H%m%S") + '.pkl', 'wb') as f:
        pickle.dump([F, dx, h, resol, vars(co)], f)
    
    plt.plot(F)

def read_line(filename):
    # read from file
    with open(savepath + filename, 'rb') as f:
        F, dx, h, resol, co_vars = pickle.load(f)
    print(co_vars) # for parameters
    plt.plot(F)

if __name__ == '__main__': # for multiprocess
    m = 12 # number of nodes
    N = 2 # number of particles, <= m
    
    savepath = 'ex_cubo/' # for pickle files
    
    # edges for cuboctahedron graph
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 9), (9, 2), (2, 10), (10, 3), (3, 11), (11, 4), (4, 12), (12, 1), (5, 9), (9, 6), (6, 10), (10, 7), (7, 11), (11, 8), (8, 12), (12, 5), (5, 6), (6, 7), (7, 8), (8, 5)]
    
    basis_onepart = LatticeBasis(m)
    basis = basis_onepart.wedge(N)
    
    H0 = -basis_onepart.graph_laplacian(edges, include_degree=False).extend(basis)
    B = OperatorList(basis, [ basis.hop({'i': i}) for i in range(1,m) ] ) # only up to m-1 to keep linearly independent together with B_0 = id
    
    co = ConstraintOptimization(H0, B)
    co.MIN_ITER_MAIN = 5000
    co.MAX_ITER_MAIN = 100000 # needs sometimes more
    co.ENERGY_CONVERGENCE_ACCURACY = 1e-8 # high accuracy
    run_linear()

    #read_line('cubo-F-line-20250208-010305.pkl')

    plt.show()