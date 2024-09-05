'''
Created on 04.09.2024

@author: mage

triangle lattice example
'''
from qmodel import *
from dft import *
import matplotlib.pyplot as plt
import numpy as np

M = 3
N = 2
b_onepart = LatticeBasis(M)
b = b_onepart.wedge(N)

t = 1
next_site = lambda qn: dict(qn, i=qn['i']%M+1)

H0 = -t * b_onepart.sum(lambda qn: b.hop(next_site(qn), qn).add_adj())
R = OperatorList(b, [b.hop({'i': i+1}) for i in range(M)])
v = [1,0,0]

#H = H0 + R*v
#sol = H.eig(hermitian=True)
#print(R.expval(sol['eigenvectors'][0], transform_real=True)) # dens

E = EnergyFunctional(H0, R)
sol = E.solve(v)
print(sol['gs_energy'])
print(R.expval(sol['gs_vector'], transform_real=True)) # dens

# # do the same using graph_laplacian
# H0_onepart = -b_onepart.graph_laplacian({(1,2), (2,3), (1,3)}, include_degree=False)
# H_onepart = H0_onepart + b_onepart.graph_pot({1: 1, 2: 0, 3: 0})
# H = H_onepart.extend(b)
# sol = H.eig(hermitian=True)
# print(sol['eigenvalues'][0])
# print(sol['eigenvectors'][0].trace(b_onepart)) # dens from trace

r_num = 10 #100 for higher resolution
r_span = np.linspace(0,1,r_num)

F_arr = np.zeros((r_num, r_num))
for i1,i2 in itertools.combinations_with_replacement(range(r_num), 2): # i1<=i2
    r1 = 1-r_span[i1]
    r2 = r_span[i2]
    r3 = N - r1 - r2
    F_arr[i1,i2] = E.legendre_transform([r1, r2, r3])['F']

fig, ax = plt.subplots() 
ax.set_xlabel(r'$\rho_1$')
ax.set_ylabel(r'$\rho_2$')
F_heatmap = ax.imshow(F_arr, cmap='Blues', interpolation='none', extent=[0,1,0,1])
#F_heatmap.set_clim(vmin=0, vmax=10)
fig.colorbar(F_heatmap)
plt.show()