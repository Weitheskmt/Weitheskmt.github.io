"""Simulations related to the adiabatic approximation"""

import numpy as np
import scipy.sparse
import scipy.linalg as la

from timeprop import propagate

import matplotlib.animation
import matplotlib.pyplot as plt


def make_adiabatic_potential_animation(potential, L=200):

    x = np.arange(L)
    ham, pot = _make_hamiltonian(L, potential, 0)
    psi_0 = la.eigh(ham.todense())[1][:, 0]

    psis = []
    pots = []
    psi = psi_0.copy()
    for n in range(300):
        ham, pot = _make_hamiltonian(L, potential, n)
        if n%3 == 0:
            pots.append(pot)
            psis.append(psi)
        psi = propagate(ham, psi, 10)
    
    fig, ax = plt.subplots(figsize=(8,3))
    plt.close()  # this prevents the output of an empty frame
    l1, = ax.plot(x, abs(psis[0])**2, label="$|\psi(x)|^2$")
    l2, = ax.plot(x, pots[0], label="$V(x)$")
    ax.set_xlabel("$x$")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.05)

    def animate(i):
        l1.set_data(x, abs(psis[i])**2)
        l2.set_data(x, pots[i])
        return (l1, l2)

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(psis), interval=50)
    
    return anim

def _make_hamiltonian(L=100, pot_func=None, time=0, t=1):
    
    ham = np.zeros(shape=(L, L), dtype=complex)
    
    if pot_func is not None:
        pot = np.array([pot_func(i, time) for i in range(L)], dtype=float)
    else:
        pot = np.zeros(shape=(L,), dtype=float)

    np.fill_diagonal(ham, 2 * t + pot)
    
    offdiag = np.zeros(shape=(L-1,), dtype=complex)
    offdiag[:] = -t
    np.fill_diagonal(ham[1:, :-1], offdiag)
    np.fill_diagonal(ham[:-1, 1:], offdiag)
    
    # finally, periodic boundary conditions
    ham[0, -1] = -t
    ham[-1, 0] = -t
    
    return scipy.sparse.csr_matrix(ham), pot