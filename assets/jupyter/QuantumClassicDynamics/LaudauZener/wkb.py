"""Simulations related to the WKB approximation"""

import numpy as np
import scipy.sparse
from cmath import exp

from timeprop import propagate

import matplotlib.animation
import matplotlib.pyplot as plt

def _make_hamiltonian(L=100, pot_func=None):
    t = 1
    
    ham = np.zeros(shape=(L, L), dtype=complex)
    
    if pot_func is not None:
        pot = np.array([pot_func(i) for i in range(L)], dtype=float)
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


def _init_wave_packet(L, zero_pos, width, energy):
    psi = np.zeros(shape=(L,), dtype=complex)
    
    for i in range(L):
        x = i
        psi[i] += exp(1j*np.sqrt(energy)*x) *exp(-0.5*(x-zero_pos)**2/width**2)
    return psi


def make_wave_packet_animation(L, pot_func, zero_pos, width, energy):
    """Simulate the time propagation of a wave packet in the potential
       given by `pot_func`
    """
    
    ham, pot = _make_hamiltonian(L, pot_func)
    init_psi = _init_wave_packet(L, zero_pos, width, energy)
    
    x = np.arange(L)
    
    psis = []
    psi = init_psi.copy()
    for n in range(1000):
        if n%10 == 0:
            psis.append(psi)
        psi = propagate(ham, psi, 10)
        
    fig, ax = plt.subplots()
    plt.close()  # this prevents the output of an empty frame
    l, = ax.plot(x, psis[0].real + energy*30, label="$\mathrm{Re}[\psi(x)]$")
    ax.plot(pot*30, label="$V(x)$")
    ax.set_xlabel("$x$")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.75, 1.75)
    
    def animate(i):
        l.set_data(x, psis[i].real + energy*30)
        return (l,)

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(psis), interval=50)

    return anim

### Code related to WKB wave functions and the connection formulas

from scipy.special import airy
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.optimize import brentq

def _calc_psi_wkb(x, x_t, m, E, V):

    def p(x):
        if E > V(x):
            return np.sqrt(2 * m * (E - V(x)))
        else:
            return np.sqrt(2 * m * (V(x) - E))
            
    if E > V(x):
        return 2/np.sqrt(p(x)) * np.sin(quad(p , x, x_t)[0] + np.pi/4)
    else:
        return 1/np.sqrt(p(x)) * np.exp(-quad(p, x_t, x)[0])
    
calc_psi_wkb = np.vectorize(_calc_psi_wkb)

def plot_patching_region_plt(V, Erange, start, stop, m, wf_scaling_factor,  y_range):

    xs = np.linspace(start, stop, 1001)
    psis_wkb = []
    airy_funs = []
    Energies = []
    xs_airys = []
    Energyvaluestr = []
    
    for E in Erange:
        x_t = brentq(lambda x: V(x) - E, start, stop)
        xs_airy = np.linspace(max(start, x_t-5), min(stop, x_t+5))
        xs_airys.append(xs_airy)
    
        alpha = (2 * m * derivative(V, x_t, dx=(stop-start)/1e7))**(1.0/3.0)
        airy_fun = np.sqrt(4 * np.pi /alpha) * airy(alpha*(xs_airy-x_t))[0] + E
        airy_funs.append(airy_fun)
        
        Energy = E
        Energies.append(Energy)
        
        energystr = str(E)
        Energyvaluestr.append(energystr)

        psi_wkb = calc_psi_wkb(xs, x_t, m, E, V)+E
        psis_wkb.append(psi_wkb)
        
    #initialize a figure in which the graph will be plotted
    fig, ax = plt.subplots()
    # initializing a line variable
    plt.close()
    line1, = ax.plot(xs, psis_wkb[0]*wf_scaling_factor, 'b-' ,label="WKB wavefunction")
    line2, = ax.plot(xs_airy, airy_funs[0]*wf_scaling_factor, 'r--', label="Airy function")
    ax.plot(xs, V(xs), 'k-' , label="V(x)")
    line3, = ax.plot([start, stop], [Energies[0], Energies[0]], 'ko--' ,label="Energy")
    envalue = ax.text(0.02, 0.02, '' , fontsize = 16, horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.set_xlabel("$z$")
    ax.legend(loc="lower right", fontsize=16)
    ax.set_xlim(start, stop)
    ax.set_ylim(y_range)
    # ax.grid()
    

    def animate(i):
        line1.set_data(xs, psis_wkb[i]*wf_scaling_factor)
        line2.set_data(xs_airys[i], airy_funs[i]*wf_scaling_factor)
        line3.set_data([start, stop], [Energies[i], Energies[i]])
        envalue.set_text('Energy: ' + Energyvaluestr[i])
        return (line1, line2, line3, envalue,)

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(psis_wkb), interval=50)

    return anim

def _wkb_free(x, m, E, V):

    def p(x):
        return np.sqrt(2 * m * (E - V(x)))

    return 2/np.sqrt(p(x)) * np.sin(quad(p, 0, x)[0] + np.pi/4)

wkb_free = np.vectorize(_wkb_free)

def pot_f(A):
    def V(x):
        return A*np.exp(-(x-1500.0)**2/400**2)
    return V

def wkb_static_animation(x, E=1.5):

    fig, ax = plt.subplots()
    plt.close()

    mags = np.linspace(-1.4, 1.4, 50)
    V = pot_f(A=mags[0])
    pot, = ax.plot(x, V(x), label=r'$V(x)$', color='black')
    wkb, = ax.plot(x, np.real(wkb_free(x, m=0.01, E=E, V=V))/15+E, label=r'$\psi(x)$', color='red')
    ax.set_xticks([]);
    ax.set_yticks([]);

    def animate(i):
        V = pot_f(A=mags[i])
        wkb.set_data(x, np.real(wkb_free(x, m=0.01, E=E, V=V))/15+E)
        pot.set_data(x, V(x))
        ax.set_ylim(-E, E+1)
        ax.legend()
        return wkb, pot,

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(mags), interval=50)

    return anim

def _make_hamiltonian_tunel(L=100, pot_func=None):
    t = 1
    
    ham = np.zeros(shape=(L, L), dtype=complex)
    
    if pot_func is not None:
        pot = np.array([pot_func(i) for i in range(L)], dtype=float)
    else:
        pot = np.zeros(shape=(L,), dtype=float)

    np.fill_diagonal(ham, 2*t+pot)
    
    offdiag = np.zeros(shape=(L-1,), dtype=complex)
    offdiag[:] = -t
    np.fill_diagonal(ham[1:, :-1], offdiag)
    np.fill_diagonal(ham[:-1, 1:], offdiag)
    
    return scipy.sparse.csr_matrix(ham), pot

def tunnel_animation(mags, potential_f):
    
    pots = []
    wfs = []

    for mag in mags:
        V = potential_f(mag)
        ham, pot = _make_hamiltonian_tunel(pot_func=V)
        evals, evec = scipy.sparse.linalg.eigsh(ham, which='SM')
        wfs.append(np.abs(evec.T[4]**2))
        pots.append(pot)

    fig, ax = plt.subplots()
    plt.close()
    x = np.linspace(0, 100, 100)
    pot, = ax.plot(x, pots[0]*10, label=r'$V(x)$', color='black')
    wf, = ax.plot(x, wfs[0], label=r'$\psi(x)$', color='red')
    #ax.set_xticks([]);
    #ax.set_yticks([]);
    ax.set_ylim(min(wfs[-1]), max(wfs[-1])+0.01)

    def animate(i):
        pot.set_data(x, pots[i])
        wf.set_data(x, wfs[i])
        ax.legend()
        return wf, pot,

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(mags), interval=100)

    return anim