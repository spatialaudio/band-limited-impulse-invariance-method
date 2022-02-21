"""
Impulse response of a first-order section with a real pole.

- Full-band impulse response (decaying exponential)
- Band-limited impulse response (BLEX function)
- The difference between the above two (BLEX residual)

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from util import (set_rcParams, decaying_exponential,
                  bandlimited_decaying_exponential)

# Continuous-time system (single-pole)
r = 1  # residue
p = -2*np.pi*2000 + 0j  # pole in the Laplace domain

# Discrete-time system setup
fs = 48000.
Ts = 1/fs
N_os = 50  # oversampling factor for quasi continuous time
fs_os = fs * N_os

# Time axes
L = 19  # FIR length
M = 9  # pre-delay (group delay of the FIR part)
t = (np.arange(L) - M) / fs  # discrete time axis
t_os = (np.arange(L*N_os) - M*N_os) / fs_os  # continuous time axis
unitstep = 1. * (t_os >= 0)

# Impulse responses (full-band and band-limited)
h_fb_os = decaying_exponential(r, p, t_os).real
h_fb = decaying_exponential(r, p, t).real
h_bl_os = bandlimited_decaying_exponential(r, p, t_os, fs).real
h_bl = bandlimited_decaying_exponential(r, p, t, fs).real
res_os = bandlimited_decaying_exponential(r, p, t_os, fs, residual=True).real
res = bandlimited_decaying_exponential(r, p, t, fs, residual=True).real


# Plots
fig_name = 'blex-and-residual-fos'
set_rcParams()
xlim = t.min()/Ts + 0.5, t.max()/Ts - 0.5
ylim = -0.7, 1.2
kw_subplots = dict(figsize=(8, 2), ncols=3, sharex=True, sharey=True,
                   gridspec_kw=dict(wspace=0.25))
kw_a = dict(lw=0.5, c='k')
kw_d = dict(marker='o', ms=6, ls='', mew=0, mfc='0.6', zorder=1)
kw_savefig = dict(dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(**kw_subplots)
ax[0].plot(t_os/Ts, h_bl_os, **kw_a)
ax[1].plot(t_os/Ts, h_fb_os, **kw_a)
ax[2].plot(t_os/Ts, res_os, **kw_a)
ax[0].plot(t/Ts, h_bl, **kw_d)
ax[1].plot(t/Ts, h_fb, **kw_d)
ax[2].plot(t/Ts, res, **kw_d)
for axi in ax:
    axi.grid(True)
    axi.set_xlim(xlim)
    axi.set_ylim(ylim)
    axi.xaxis.set_major_locator(MultipleLocator(5))
    axi.xaxis.set_minor_locator(MultipleLocator(1))
    axi.set_xlabel('$t$ / $T$')
ax[0].set_title('BLEX')
ax[1].set_title('Full-band IR')
ax[2].set_title('BLEX residual')
ax[0].text(10, 0.25, r'$=$', va='center')
ax[1].text(10, 0.25, r'$+$', va='center')
plt.savefig(fig_name + '.pdf', **kw_savefig)
plt.savefig(fig_name + '.png', **kw_savefig)
