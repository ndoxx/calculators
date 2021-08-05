#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from numpy.core.numeric import NaN


"""
This program computes precisely the helical auto-inductance of an air coil.
This program is a direct transcription of my helicoils.m Matlab helical auto-inductance
calculator, based on Robert Weaver's code (in open office basic, 2012), itself based on 
a paper from Chester Snow (1926).
The integral is evaluated using a numerical method that uses the composite Simpson's rule
as well as Newton-Raphson iterations.
"""

K_MAX_ERROR = 1e-4

class GMDApprox(Enum):
    ROUND_CROSS_SECTION = 1
    LOW_FREQ = 2
    HIGH_FREQ = 3


def helical_inductance(Lw, dw, psi, r, max_error=K_MAX_ERROR, gmd_approx: GMDApprox = GMDApprox.LOW_FREQ):
    """
    Uses helical filament mutual inductance formula evaluated using
    Simpson's rule, and conductor gmd
        -----------------------------
        Lw = total length of wire
        psi = pitch angle of winding
        r = radius of winding
        dw = wire diameter
        max_error = max allowable error
        gmd_approx = approximation to use for the conductor gmd
        -----------------------------
    """

    # If Lw > 2*pi*r, check that pitch angle >= close wound pitch
    loop_len = 2*np.pi*r
    if Lw > loop_len:
        close_wound_pitch = np.arcsin(dw / loop_len)
        if psi < close_wound_pitch:
            raise Exception('Pitch angle is too small')

    # Geometric mean distance of solid round conductor
    if gmd_approx == GMDApprox.ROUND_CROSS_SECTION:
        # Round cross section approx
        g = 0.5 * np.exp(-0.25) * dw
    elif gmd_approx == GMDApprox.LOW_FREQ:
        # Solid conductor, low frequency
        g = 0.25 * np.exp(-0.25) * (1+1/np.cos(psi)) * dw
    else:
        # Thin tubular conductor (skin effect), high frequency
        g = 0.25 * (1+1/np.cos(psi)) * dw

    # Calculate Filament 2 offset angle
    # Trap for psi = 0 condition in which case ThetaO = 0 and Y0 = g
    # Trap for psio = 0 condition in which case use simplified
    # formula for ThetaO and Y0 = 0
    # which happens with circular(non-helical) filament
    rr = r**2
    psi0 = 0.5 * np.pi - psi
    if psi == 0:
        Theta0 = 0
        Y0 = g
    elif psi0 == 0:
        cosThetaO = 1 - (g**2/(2*rr))
        ThetaO = -np.abs(np.atan(np.sqrt(1-cosThetaO**2)/cosThetaO))
        Y0 = 0
    else:
        # Use Newton-Raphson method
        k1 = g**2/(2*rr)-1
        k2 = np.tan(psi0)
        k2 = 0.5*k2**2
        t1 = g/r*np.sin(psi)

        while True:
            t0 = t1
            t1 = t0-(k1+np.cos(t0)-k2*t0**2)/(-np.sin(t0)-2*k2*t0)
            if np.abs(t1-t0) < 1e-12:
                break

        ThetaO = -np.abs(t1)
        # Calculate Filament 2 Y-offset, using formula (29)
        Y0 = np.sqrt(g**2-2*rr*(1-np.cos(ThetaO)))

        # Psi constants
        c2s = np.cos(psi)**2
        ss = np.sin(psi)
        k = np.cos(psi)/r

    # --- Start of Simpson's Rule code ---
    # Integrand to be summed with Simpson
    def integrand(phi, kphitheta, sinpsi, cos2psi, rr, y):
        return (1+cos2psi*(np.cos(kphitheta)-1))/np.sqrt(2*rr*(1-np.cos(kphitheta))+np.power((sinpsi*phi-y), 2))

    # Initial state
    grandtotal = 0
    a = 0
    b = Lw/32768
    if b > Lw:
        b = Lw

    # Let's go
    while True:
        dx = b-a
        m = 1
        CurrentErr = 2*max_error
        kat = k*a
        kbt = k*b
        Sum2 = (Lw-a)*(integrand(-a, -kat-ThetaO, ss, c2s, rr, Y0)+integrand(a, kat-ThetaO, ss, c2s, rr, Y0)) + \
               (Lw-b)*(integrand(-b, -kbt-ThetaO, ss, c2s, rr, Y0)+integrand(b, kbt-ThetaO, ss, c2s, rr, Y0))
        # Initialize LastResult to trapezoidal area for termination test purposes
        LastIntg = Sum2/2*dx

        while True:
            m = 2*m
            dx = dx/2
            Sum = 0
            SumA = 0

            for ii in range(1, m, 2):
                phi = ii*dx+a
                kpt = k*phi
                Sum = Sum + (Lw-phi)*(integrand(-phi, -kpt-ThetaO, ss, c2s, rr, Y0)+integrand(phi, kpt-ThetaO, ss, c2s, rr, Y0))

            Integral = (4*Sum+Sum2)*dx/3
            CurrentErr = abs((Integral)/(LastIntg)-1)
            LastIntg = Integral
            Sum2 = Sum2+Sum*2

            if CurrentErr <= max_error or m >= 512:
                break

        grandtotal = grandtotal+Integral
        a = b
        b = b*2
        if b > Lw:
            b = Lw

        if a >= Lw:
            break

    return 1e-7 * grandtotal


def stretch_test():
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

        L = helical_inductance(Lw, dw, psi, r, maxerr, GMDApprox.LOW_FREQ)
        Lapp = 1e-4*0.394*r**2*Neff**2/(9*r+10*Neff*p)
        Lh.append(L)
        Lh_app.append(Lapp)

    # Plot the results
    Lh = np.array(Lh)
    Lh_app = np.array(Lh_app)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(1e3*prange, 1e9*Lh, linewidth=2, color='blue')
    axs[0].set_title('Inductance of a coil being stretched till it\'s straight')
    axs[0].set_xlabel('Pitch (mm)')
    axs[0].set_ylabel('Inductance (nH)')

    axs[1].plot(1e3*prange[0:40], 1e9*Lh[0:40], linewidth=2, color='blue', label='Helical inductance')
    axs[1].plot(1e3*prange[0:40], 1e9*Lh_app[0:40], linewidth=2, color='red', label='Approx. inductance')
    axs[1].set_title('Inductance formulae comparison')
    axs[1].set_xlabel('Pitch (mm)')
    axs[1].set_ylabel('Inductance (nH)')
    axs[1].legend()

    plt.show()


if __name__ == '__main__':
    stretch_test()
