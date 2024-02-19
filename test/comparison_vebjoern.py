import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def HamiltonianSp(t,l,v,j,n):
    _H = sp.zeros(2*n,2*n)
    for i in range(n):
        for k in range(n):
            if i == k:
                _H[2*i,2*k] = v + 1/2 + i
                _H[2*i+1,2*k+1] = -v + 1/2+i
                _H[2*i,2*k+1] = -t
                _H[2*i+1,2*k] = -t
            elif i == k+1:
                _H[2*i,2*k] = sp.sqrt(k+1)*(l+j)/sp.sqrt(2)
                _H[2*i+1,2*k+1] = sp.sqrt(k+1)/(-l+j)*sp.sqrt(2)
            elif i == k-1:
                _H[2*i,2*k] = sp.sqrt(k)*(l+j)/sp.sqrt(2)
                _H[2*i+1,2*k+1] = sp.sqrt(k)*(-l+j)/sp.sqrt(2)
            else: 
                None
    return _H

n=50 
#t,l,v,j = sp.symbols('t lambda v j')
#H = HamiltonianSp(t,l,v,j,n)
#sp.pprint(H)

def Hamiltonian(t,l,v,j,n):
    _H = np.zeros((2*n,2*n))
    for i in range(n):
        for k in range(n):
            if i == k:
                _H[2*i,2*k] = v + 1/2 + i
                _H[2*i+1,2*k+1] = -v + 1/2+i
                _H[2*i,2*k+1] = -t
                _H[2*i+1,2*k] = -t
            elif i == k+1:
                _H[2*i,2*k] = sp.sqrt(k+1)*(l+j)/np.sqrt(2)
                _H[2*i+1,2*k+1] = sp.sqrt(k+1)*(-l+j)/np.sqrt(2)
            elif i == k-1:
                _H[2*i,2*k] = sp.sqrt(k)*(l+j)/np.sqrt(2)
                _H[2*i+1,2*k+1] = sp.sqrt(k)*(-l+j)/np.sqrt(2)
            else: 
                None
    return _H

def x(psi):
    up = psi[::2]
    down = psi[1::2]
    if len(np.shape(psi)) == 1: 
        _x = 0
        for i in range(len(up)-1):
            _x += up[i]*up[i+1] * np.sqrt((i+1)/2) +  up[i+1]*up[i]*np.sqrt((i+1)/2)
            _x += down[i]*down[i+1] * np.sqrt((i+1)/2) + down[i+1]*down[i]*np.sqrt((i+1)/2)
        return _x
    else:
        _x = np.zeros(len(psi[0,:]))
        for i in range(len(up[:,0])-1):
            _x[:] += up[i,:]*up[i+1,:] * np.sqrt((i+1)/2) +  up[i+1,:]*up[i,:]*np.sqrt((i+1)/2)
            _x[:] += down[i,:]*down[i+1,:] * np.sqrt((i+1)/2) + down[i+1,:]*down[i,:]*np.sqrt((i+1)/2)
        return _x



def sigma(psi):
    up = psi[::2]
    down = psi[1::2]
    return np.sum(up**2,axis=0) - np.sum(down**2,axis=0)

t=1
l=5
v=0
j=0

H = Hamiltonian(t,l,v,j,n)
eigvals, eig = np.linalg.eigh(H)
#eig /= np.linalg.norm(eig, axis=0)**2

print(f'x = {x(eig[:,0]):.3E}')
print(f'sigma_z = {sigma(eig[:,0]):.3E}')

fig0, ax0 = plt.subplots(2, figsize=(9,5),sharex=True, sharey=True)
fig0.suptitle(f'State distribution \nfor $\lambda$={l}, t={t}, v={v}, j={j}')

ax0[1].set_xlabel(r'$n$')
ax0[0].set_ylabel(r'$P_\uparrow (n)$')
ax0[1].set_ylabel(r'$P_\downarrow (n)$')

ax0[0].bar(np.arange(0,n),eig[::2,0]**2)
ax0[1].bar(np.arange(0,n),eig[1::2,0]**2)
fig0.tight_layout()


t = 1
l = np.linspace(0,10,1001)
v = 0
j = 0  


E0 = np.zeros(len(l))
psi0 = np.zeros((2*n,len(l)))

for i in range(len(l)):
    H = Hamiltonian(t,l[i],v,j,n)
    eigvals, eig = np.linalg.eigh(H)
    E0[i] = eigvals[0]
    psi0[:,i] = eig[:,0]



fig01, ax01 = plt.subplots(1, figsize=(9,5))
ax01.set_xlabel(r'$\lambda$')
ax01.set_ylabel(r'$E_0$')
ax01.set_title(f'Ground state energy \nfor n={n}, t={t}, v={v}, j={j}')
ax01.plot(l,E0)
fig01.tight_layout()


fig02, ax02 = plt.subplots(1,2, figsize=(9,5),sharex=False, sharey=True)
ax02[0].set_xlabel(r'$\lambda$')
ax02[1].set_xlabel(r'$\lambda$')
ax02[0].set_ylabel(r'$P_\uparrow(n)$')
ax02[1].set_ylabel(r'$P_\downarrow(n)$')
fig02.suptitle(f'n={n}, t={t}, v={v}, j={j}')

for i in range(10):
    ax02[0].plot(l,psi0[2*i,:]**2,label=f'n={i}')
    ax02[1].plot(l,psi0[2*i+1,:]**2)

fig02.legend()
fig02.tight_layout()

fig03, ax03 = plt.subplots(2, figsize=(9,5),sharex=True,sharey=False)
fig03.suptitle(f'n={n}, t={t}, v={v}, j={j}')

ax03[1].set_xlabel(r'$\lambda$')
ax03[0].set_ylabel(r'$\sigma$')
ax03[1].set_ylabel(r'$x$')

ax03[0].plot(l,sigma(psi0))
ax03[1].plot(l,x(psi0))
fig03.tight_layout()




'''
fig3, ax3 = plt.subplots(1, figsize=(9,5))
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel(r'$E_0$')
ax3.set_title(f'Ground state distribution \nfor n={n}, t={t}, v={v}, j={j}')
ax3.plot(l,np.linalg.norm(psi0[::2,:],axis=0)**2, label=r'$\psi_-$')
ax3.plot(l,np.linalg.norm(psi0[1::2,:],axis=0)**2,label=r'$\psi_+$')
fig3.legend()
fig3.tight_layout()


t = 1
l = 1
v = np.linspace(-10,10,101)
j = 100  

E0 = np.zeros(len(v))
psi0 = np.zeros((2*n,len(v)))

for i in range(len(v)):
    H = Hamiltonian(t,l,v[i],j,n)
    eigvals, eig = np.linalg.eigh(H)
    E0[i] = eigvals[0]
    psi0[:,i] = eig[:,0]


fig4, ax4 = plt.subplots(1, figsize=(9,5))
ax4.set_xlabel(r'$v$')
ax4.set_ylabel(r'$E_0$')
ax4.set_title(f'Ground state energy \nfor n={n}, t={t}, $\lambda$={l}, j={j}')
ax4.plot(v,E0)
fig4.tight_layout()


fig5, ax5 = plt.subplots(1,2, figsize=(9,5),sharex=False, sharey=True)
ax5[0].set_xlabel(r'$v$')
ax5[1].set_xlabel(r'$v$')
ax5[0].set_ylabel(r'$P(n)$')
ax5[0].set_title('Site 1')
ax5[1].set_title('Site 2')

for i in range(10):
    ax5[0].plot(v,psi0[2*i,:]**2,label=f'n={i}')
    ax5[1].plot(v,psi0[2*i+1,:]**2)

fig5.legend()
fig5.tight_layout()

fig6, ax6 = plt.subplots(1, figsize=(9,5))
ax6.set_xlabel(r'$v$')
ax6.set_ylabel(r'$P(v)$')
ax6.set_title(f'Ground state distribution \nfor n={n}, t={t}, $\lambda$={l}, j={j}')
ax6.plot(v,np.linalg.norm(psi0[::2,:],axis=0)**2, label=r'$\psi_-$')
ax6.plot(v,np.linalg.norm(psi0[1::2,:],axis=0)**2,label=r'$\psi_+$')
fig6.legend()
fig6.tight_layout()

t = 1
l = 1
v = 10
j = 10 
n = np.arange(1,200)
E0 = np.zeros(len(n))

for i in range(len(n)):
    H = Hamiltonian(t,l,v,j,n[i])
    eigvals, eig = np.linalg.eigh(H)
    E0[i] = eigvals[0]

fig7, ax7 = plt.subplots(1, figsize=(9,5))
ax7.set_xlabel(r'$n$')
ax7.set_ylabel(r'$-E_0$')
ax7.set_title(f'Ground state energy \nfor $\lambda$={l}, t={t}, v={v}, j={j}')
ax7.set_xscale('log')
ax7.plot(n,E0)
#ax7.plot(n,np.linalg.norm(psi0[::2,:],axis=0)**2, label=r'$\psi_-$')
#ax7.plot(l,np.linalg.norm(psi0[1::2,:],axis=0)**2,label=r'$\psi_+$')

fig7.tight_layout()
'''


plt.show()
