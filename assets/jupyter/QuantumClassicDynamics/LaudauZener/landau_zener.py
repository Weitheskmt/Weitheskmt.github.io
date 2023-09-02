import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from timeprop import propagate
import numpy as np

def make_hamiltonian(L=100, pot_func=None, time=0, t=1):
    # t is gap
    ham = np.zeros(shape=(L, L), dtype=complex)

    if pot_func is not None:
        pot = np.array([pot_func(i, time) for i in range(L)], dtype=float)
    else:
        pot = np.zeros(shape=(L,), dtype=float)

    np.fill_diagonal(ham, pot)

    offdiag = np.zeros(shape=(L-1,), dtype=complex)
    offdiag[:] = -t
    np.fill_diagonal(ham[1:, :-1], offdiag)
    np.fill_diagonal(ham[:-1, 1:], offdiag)

    return ham, pot

def landau_zener_data(v, L=3, npts=100, gap=1):
    """
    Evolve a wavefunction over an adiabatic Hamiltonian of size L x L at velocity v
    """

    if L == 2:
        def pot_r(i, t):
            return t*v*(-1)**i
        t0 = -5/v
        tf = 5/v
    else:
        def pot_r(i, t):
            return (-t*v+2*i)*(-1)**i
        t0 = -5/v
        tf = 8/v


    step = (tf-t0)/npts
    times = np.arange(t0, tf, step)
    psi = np.zeros(L)
    psi[0] = 1

    ens = []
    wfs = []
    wfs.append(psi)

    for time in times:
        ham = make_hamiltonian(L=L, pot_func=pot_r, time=time, t=gap)[0]
        psi = propagate(ham, psi, step)
        en, _ = np.linalg.eigh(ham)
        ens.append(en)
        wfs.append(psi)

    wfs = np.array(wfs)
    ens = np.array(ens)
    probs = np.abs(wfs)

    return times, ens, probs

def animate_landau_zener_2plots(times, ens, probs, interval, title, subtitles=None):
    """
    Generate animation of two systems with different velocities
    """

    labels=[r'$|c_0(t)|^2$', r'$|c_1(t)|^2$', r'$|c_2(t)|^2$']
    colors=['darkblue', 'red', 'green']

    size = 200

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),dpi=350)
    plt.close()
    l1 = len(ens[0][0])
    l2 = len(ens[1][0])
    scatter1 = axes[0].scatter(times[0]*np.ones(l1), ens[0][0, :], s=size, c=colors[:l1])
    scatter2 = axes[1].scatter(times[0]*np.ones(l2), ens[1][0, :], s=size, c=colors[:l2])
    fig.suptitle(title)

    k = 0
    for ax in axes:
        j = 0
        L = len(ens[k][0])
        #print(colors[:L])
        for level in ens[k].T:
            ax.plot(times, level, c=colors[:L][j])
            j += 1
        ytop = ens[k][0, -1]
        ybot = ens[k][0, 0]
        ax.set_ylim(ybot, ytop);

        ax.set_xlabel('time')
        ax.set_ylabel('Energy')

        for i in range(L):
            ax.scatter(0, 1000, s=size, label=labels[:L][i], c=colors[:L][i])

        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.legend(fontsize=12)

        if subtitles is not None:
            ax.set_title(subtitles[k])
        k += 1

    def update(i):
        ts1 = times[i]*np.ones(len(ens[0][0]))
        ts2 = times[i]*np.ones(len(ens[1][0]))
        es1 = ens[0][i, :]
        es2 = ens[1][i, :]
        scatter1.set_offsets(np.vstack([ts1, es1]).T)
        scatter1.set_sizes(probs[1][i]**2*size)
        scatter2.set_offsets(np.vstack([ts2, es2]).T)
        scatter2.set_sizes(probs[0][i]**2*size)
        return scatter1, scatter2,

    anim = FuncAnimation(fig, update, interval=interval)

    return anim


def animate_landau_zener(times, ens, probs, L, interval, title):
    """
    Generate single animation
    """

    labels=[r'$|c_0(t)|^2$', r'$|c_1(t)|^2$', r'$|c_2(t)|^2$']
    colors=['darkblue', 'red', 'green']
    labels=labels[:L]
    colors=colors[:L]
    size = 200

    fig, ax = plt.subplots(figsize=(5, 5),dpi=350)
    plt.close()
    j = 0

    for level in ens.T:
        ax.plot(times, level, c=colors[j])
        j += 1
    ax.set_xlabel('time')
    ax.set_ylabel('Energy')

    scatter = ax.scatter(times[0]*np.ones(L), ens[0, :], s=probs[0, :]**2*size, c=colors);

    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_title(title);

    ytop = ens[0, -1]
    ybot = ens[0, 0]
    ax.set_ylim(ybot, ytop);

    ax.set_xlabel('time')
    ax.set_ylabel('Energy')

    for i in range(L):
        ax.scatter(0, 1000, s=size, label=labels[:L][i], c=colors[:L][i])

    ax.legend(fontsize=12);

    def update(i):

        ts = times[i]*np.ones(L)
        es = ens[i, :]
        scatter.set_offsets(np.vstack([ts, es]).T);
        scatter.set_sizes(probs[i]**2*size);
        return scatter,

    anim = FuncAnimation(fig, update, interval=interval);

    return anim
