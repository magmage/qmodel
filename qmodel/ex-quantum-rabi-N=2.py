'''
Created on 10.08.2024

@author: mage

Dicke model example
M = 1, N = 2
'''
from qmodel import *
from dft import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

oscillator_size = 5
b_oscillator = NumberBasis(oscillator_size)
# define qubits separately so also operators can act separately on them
b_spin1 = SpinBasis('s1')
b_spin2 = SpinBasis('s2')
b = b_oscillator.tensor(b_spin1.tensor(b_spin2))

print(b)

num_op = b_oscillator.diag().extend(b) # number operator for oscillator
x_op = b_oscillator.x_operator().extend(b)
sigma_z1 = b_spin1.sigma_z().extend(b) # also possible as vec
sigma_z2 = b_spin2.sigma_z().extend(b)
sigma_x1 = b_spin1.sigma_x().extend(b)
sigma_x2 = b_spin2.sigma_x().extend(b)

t = 1
H0_Rabi_KS = 2*(num_op + 1/2) - t*(sigma_x1 + sigma_x2)
CouplingRabi = x_op*(sigma_z1 + sigma_z2)

lam = 1
H = H0_Rabi_KS + lam*CouplingRabi

v_num = 101
v_max = 10
v_space = np.linspace(-v_max,v_max,v_num)
E_arr = np.zeros((v_num, v_num))
N_arr = np.zeros((v_num, v_num))

fig, ax = plt.subplots(1,2) 
E_heatmap = ax[0].imshow(E_arr, cmap='hot', interpolation='none', extent=[-v_max,v_max,-v_max,v_max])
N_heatmap = ax[1].imshow(N_arr, cmap='Blues', interpolation='none', extent=[-v_max,v_max,-v_max,v_max])
#E_heatmap.set_clim(vmin=-20, vmax=0)
#N_heatmap.set_clim(vmin=0, vmax=1)
#fig.colorbar(E_heatmap)
#fig.colorbar(N_heatmap)

def plot(lam):
    H0_Rabi = H0_Rabi_KS + lam*CouplingRabi
    E = EnergyFunctional(H0_Rabi, [sigma_z1, sigma_z2])
    for i1,i2 in itertools.product(range(v_num), repeat=2):
        gs = E.solve([v_space[i1],v_space[i2]])
        #dens = [sigma_z1.expval(gs['gs_vector']), sigma_z2.expval(gs['gs_vector'])]
        E_arr[i1,i2] = gs['gs_energy']
        N_arr[i1,i2] = num_op.expval(gs['gs_vector'], transform_real=True)
    E_heatmap.set_data(E_arr)
    N_heatmap.set_data(N_arr)
    ax[0].set_title(r'$\lambda = {:.2f}$'.format(lam))

plot(lam=10)
E_heatmap.set_clim(vmin=np.min(E_arr), vmax=np.max(E_arr))
N_heatmap.set_clim(vmin=np.min(N_arr), vmax=np.max(N_arr))
fig.colorbar(E_heatmap)
fig.colorbar(N_heatmap)

# F plot (slow!)
lam = 0
H0_Rabi = H0_Rabi_KS + lam*CouplingRabi
E = EnergyFunctional(H0_Rabi, [sigma_z1, sigma_z2])
s_num = 100
s_span = np.linspace(-1,1,s_num)

F_arr = np.zeros((s_num, s_num))
for i1,i2 in itertools.product(range(s_num), repeat=2):
    s1 = s_span[i1]
    s2 = s_span[i2]
    F_arr[i1,i2] = E.legendre_transform([s1, s2])['F']

fig2, ax2 = plt.subplots(1,2) 
F_heatmap = ax2[0].imshow(F_arr, cmap='hot', interpolation='none', extent=[-1,1,-1,1])
fig2.colorbar(F_heatmap)

ax2[1].plot(s_span, F_arr[round(s_num/2),:])

plt.show()

# frames_num = 100
# lam_span = np.linspace(0,10,frames_num)
# def frame(i):
#     print('generating frame {} / {}'.format(i+1, frames_num))
#     plot(lam_span[i])
#     return [E_heatmap,N_heatmap]
#
# fps = 10
# anim = animation.FuncAnimation(
#                                fig, 
#                                frame, 
#                                frames = frames_num,
#                                interval = 1000/fps, # in ms
#                                )
# plt.rcParams['animation.ffmpeg_path']='C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe'
# anim.save('test_anim.mp4', fps=fps)