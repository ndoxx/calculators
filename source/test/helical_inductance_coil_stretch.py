#!/usr/bin/python3

import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from inductance import helical_inductance, wire_inductance_lf, GMDApprox
import matplotlib.pyplot as plt


def main():
    # N turns
    N = 5
    # Wire diameter
    dw = 0.6e-3
    # Coil radius (conductor center to center)
    r = 1.5e-3+dw/2
    # Maximal allowable error
    maxerr = 1e-4

    Lh = []
    Lh_app = []

    # Here we stretch a coil that starts fully compressed (pitch very close to
    # the close wound pitch) till it is wtraight, and keep track of the helical
    # inductance along the way.
    # We also compute the classical approximation for the inductance, for later
    # comparison.
    # The classical approximation should vanish, while the true helical inductance
    # should approach the non-zero inductance of a straight wire.
    prange = np.arange(dw*1.01, 300e-3, 0.5e-3)
    for p in prange:
        # Pitch factor
        px = p / (2*np.pi)
        # Coil length
        Lc = N*p
        # Diameter / Length ratio
        DL = 2*r/Lc
        # Pitch angle
        psi = np.arctan(p/(2*np.pi*r))
        # Close wound pitch angle
        psic = np.arcsin(dw/(2*np.pi*r))
        psicdeg = psic*180/np.pi

        # Wire length
        # Lw = N*np.sqrt(p**2+(2*np.pi*r)**2)
        # Lw = (2*np.pi*N*r)/np.cos(psi)
        Lw = 30e-2
        Neff = Lw*np.cos(psi)/(2*np.pi*r)

        L = helical_inductance(Lw, dw, psi, r, maxerr, GMDApprox.ROUND_CROSS_SECTION)
        Lapp = 1e-4*0.394*r**2*Neff**2/(9*r+10*Neff*p)
        Lh.append(L)
        Lh_app.append(Lapp)

    # Wire inductance
    L_wire = 1e9*wire_inductance_lf(Lw, dw)

    # Plot the results
    Lh = np.array(Lh)
    Lh_app = np.array(Lh_app)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(1e3*prange, 1e9*Lh, linewidth=2, color='blue', label='Helical inductance')
    axs[0].axhline(y=L_wire, color='black', linestyle='--', label='Wire inductance')
    axs[0].set_title(
        'Inductance of a coil being stretched till it\'s straight')
    axs[0].set_xlabel('Pitch (mm)')
    axs[0].set_ylabel('Inductance (nH)')
    axs[0].legend()

    axs[1].plot(1e3*prange[0:80], 1e9*Lh[0:80], linewidth=2,
                color='blue', label='Helical inductance')
    axs[1].plot(1e3*prange[0:80], 1e9*Lh_app[0:80], linewidth=2,
                color='red', label='Approx. inductance')
    axs[1].axhline(y=L_wire, color='black',
                   linestyle='--', label='Wire inductance')
    axs[1].set_title('Inductance formulae comparison')
    axs[1].set_xlabel('Pitch (mm)')
    axs[1].set_ylabel('Inductance (nH)')
    axs[1].legend()

    plt.show()


if __name__ == '__main__':
    main()
