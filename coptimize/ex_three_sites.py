'''
Constrained Search in Imaginary Time
method for calculating F(rho) functional of 3-site lattice with 2 particles (triangle graph)
'''

from qmodel import LatticeBasis, OperatorList, timer
from coptimize import ConstraintOptimization
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt 
import pickle # io
import time

m = 3 # number of nodes
N = 2 # number of particles, <= m

savepath = 'ex_three_sites/' # for pickle files

# edges for triangle graph
edges = {(1, 2), (2, 3), (3, 1)}

basis_onepart = LatticeBasis(m)
basis = basis_onepart.wedge(N)

H0 = -basis_onepart.graph_laplacian(edges, include_degree=False).extend(basis)
B = OperatorList(basis, [ basis.hop({'i': i}) for i in range(1,m) ] ) # only up to m-1 to keep linearly independent together with B_0 = id

co = ConstraintOptimization(H0, B)
# default parameters

def plot(F):
    plt.figure()
    plt.imshow(F, cmap='managua_r', interpolation='none', vmin=-1, vmax=1)
    plt.colorbar()
    ax = plt.gca()
    ax.set_axis_off()

def barycentric(p, triangle):
    # get barycentric coordinates in 2d
    # triangle gives three points
    # from https://gamedev.stackexchange.com/questions/23743/
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = p - triangle[0]
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array((u,v,w))

@timer
def run(random_jump_method):
    print("run of triangle graph with random-phase method: " + str(random_jump_method))
    resol_x = 201 # resolution in x direction
    resol_y = int(resol_x*sqrt(3)/2)
    
    F = np.full((resol_y,resol_x), np.nan) # rows,cols; fill with NaN
    triangle = (np.array(((resol_x-1)/2, 0)), np.array((resol_x-1, resol_y-1)), np.array((0, resol_y-1)))
    
    for ix in range(resol_x):
        for iy in range(resol_y): # height of triangle
            if ix/(resol_x-1) >= (1 - iy/(resol_y-1)) / 2 and ix/(resol_x-1) <= 1 - (1 - iy/(resol_y-1)) / 2: # inside triangle
                # get rho for point in grid
                b = barycentric(np.array((ix, iy)), triangle)
                rho = b[0]*np.array((1,1,0)) + b[1]*np.array((0,1,1)) + b[2]*np.array((1,0,1))
                print("run rho = ", end="")
                print(rho)
                rho = rho[0:-1] # remove last item, fixed by normalization
                res = co.run(rho, random_jump_method=random_jump_method)
                F[iy,ix] = res['min']
                # analytical result in S_1 region
                #F[iy,ix] = 2*(sqrt((1-rho[0])*(1-rho[2]))-sqrt((1-rho[0])*(1-rho[1]))-sqrt((1-rho[1])*(1-rho[2])))
                print(F[iy,ix])
    
    plot(F)
    
    # save to file
    with open(savepath + 'three-sites-F-' + time.strftime("%Y%m%d-%H%M%S") + '_jmp' + str(random_jump_method).capitalize() + '.pkl', 'wb') as f:
        pickle.dump([F, resol_x, resol_y, edges, vars(co)], f)
        
def read(filename):
    # read from file
    with open(savepath + filename, 'rb') as f:
        F, resol_x, resol_y, edges, co_vars = pickle.load(f)
    print(co_vars) # for parameters
    plot(F)

#run(None)
#run('sign')
#run('phase')

# special example: non-uniquely v-rep density on boundary
#rho1 = 1
#rho2 = 0.5
#print(co.run([rho1, rho2], random_jump_method='phase'))

# special example: non-v-rep density on boundary (needs infinite beta) -> also converges within accuracy
#rho1 = 1
#rho2 = 0.4
#print(co.run([rho1, rho2], random_jump_method='phase'))

# plot from saved data
read('three-sites-F-20250228-102911_jmpNone.pkl')
read('three-sites-F-20250228-102911_jmpSign.pkl')
read('three-sites-F-20250228-102911_jmpPhase.pkl')

plt.show()
