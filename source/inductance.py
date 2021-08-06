import sys
import numpy as np
from enum import Enum


K_MAX_ERROR = 1e-4

class GMDApprox(Enum):
    ROUND_CROSS_SECTION = 1
    LOW_FREQ = 2
    HIGH_FREQ = 3


class Material:
    def __init__(self, name: str, conductivity: float, permeability: float):
        self.name = name
        self.conductivity = conductivity
        self.permeability = permeability


# Using https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118936160.app2
# Table B.1
K_MATERIALS = {
    'Cu': Material('copper',    5.96e7, 0.999994),
    'Au': Material('gold',      4.11e7, 0.999998),
    'Ag': Material('silver',    6.30e7, 0.99999981),
    'Al': Material('aluminium', 3.77e7, 1.00000065),
    'Ni': Material('nickel',    1.43e7, 600),
    'Fe': Material('iron',      1.00e7, 6000),
}


def wire_inductance(Lw: float, dw: float, f: float = 0, mat: str = 'Cu'):
    """
        This function computes the self-inductance of a wire.
        Source: Transmission Line Design Handbook by Brian C Wadell, Artech House 1991, paragraph 6.2.1.1
        -----------------------------
        Lw = total length of wire [m]
        dw = wire diameter [m]
        f = frequency [Hz]
        mat = material (chemical element symbol)
        return: self-inductance [H]
        -----------------------------
    """

    if not mat in K_MATERIALS:
        raise Exception('Unknown material')

    material = K_MATERIALS[mat]

    r = 0.5 * dw
    mu = 4e-7 * np.pi * material.permeability
    x = 2 * np.pi * (100*r) * np.sqrt(2 * mu * f / material.conductivity)
    T = np.sqrt((0.873011 + 0.00186128 * x) / (1 - 0.278381 * x + 0.127964 * x**2))
    q = dw / (2*Lw)
    L = 2e-7 * Lw * (np.log(2/q) - 1 + q + 0.25 * material.permeability * T)

    return L


def wire_inductance_lf(Lw: float, dw: float, mat: str = 'Cu'):
    """
        This function computes the self-inductance of a wire (low frequency approx).
        -----------------------------
        Lw = total length of wire [m]
        dw = wire diameter [m]
        mat = material (chemical element symbol)
        return: self-inductance [H]
        -----------------------------
    """

    if not mat in K_MATERIALS:
        raise Exception('Unknown material')

    material = K_MATERIALS[mat]

    q = dw / (2*Lw)
    x = np.sqrt(1 + q**2)
    # 200 nH/m is $\mu_0/2*\pi$ the permeability of free space over 2 pi
    L = 2*1e-7*Lw*(np.log((1+x)/q) - x + q + 0.25 * material.permeability)

    return L


def helical_inductance(Lw: float, dw: float, psi: float, r: float, max_error: float =K_MAX_ERROR, gmd_approx: GMDApprox = GMDApprox.ROUND_CROSS_SECTION):
    """
        This function computes precisely the helical self-inductance of an air coil.
        This function is a direct transcription of my helicoils.m Matlab helical self-inductance
        calculator, based on Robert Weaver's code (in open office basic, 2012), itself based on 
        a paper from Chester Snow (1926).
        The integral is evaluated using a numerical method that uses the composite Simpson's rule
        as well as Newton-Raphson iterations.
        -----------------------------
        Lw = total length of wire [m]
        psi = pitch angle of winding [rad]
        r = radius of winding (conductor center to center) [m]
        dw = wire diameter [m]
        max_error = max allowable error
        gmd_approx = approximation to use for the conductor gmd
        return: self-inductance [H]
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
