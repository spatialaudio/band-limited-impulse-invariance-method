"""
Digital modeling of an analog prototype filter using the impulse invariance
method.

- un-corrected
- corrected
- DC-matched
- band-limited

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import residue, freqz, freqs
from scipy.signal.windows import kaiser
from util import (db, set_rcParams, log_frequency, impulse_invariance,
                  freqz_parfilt)
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset)


# constants
c = 343
fs = 48000
fNyquist = 0.5 * fs
os = 10  # oversampling rate

# frequencies
fmin, fmax, num_f = 93.8, 24000, 2000
f = log_frequency(fmin, fmax, num_f)
w = 2*np.pi*f

# peaking EQ
fc = 0.5 * fNyquist
wc = 2*np.pi*fc
g = 10**(15/20)
Q = 1/np.sqrt(2)
b = np.array([0, 1/Q/wc, 0])
a = np.array([1/wc**2, 1/Q/wc, 1])
rpk = residue(b, a)  # partial fraction expansion (continuous-time)
k = np.asarray([0])

# Filter design
FIR_params = [(5, 2), (11, 5), (21, 10), (41, 20)]
Filters_bl = [impulse_invariance(
    *rpk, *firpar, fs, 'dcmbandlimited', kaiser(firpar[0], beta=8.6))
    for firpar in FIR_params]
Filter_uncorr = impulse_invariance(*rpk, 1, 0, fs, 'uncorrected')
Filter_corr = impulse_invariance(*rpk, 1, 0, fs, 'corrected')
Filter_dcm = impulse_invariance(*rpk, 1, 0, fs, 'dcmatched')
Filter_osdcm = impulse_invariance(*rpk, 1, 0, 10*fs, 'dcmatched')

# Frequency responses
H_a = freqs(b, a, worN=2*np.pi*f)[1]
H_bl_stack = [freqz_parfilt(Filter, f, fs, causal=False)[1]
              for Filter in Filters_bl]
H_fir_stack = [freqz(Filter[1], worN=f, fs=fs)[1] for Filter in Filters_bl]
H_uncorr = freqz_parfilt(Filter_uncorr, worN=f, fs=fs, causal=False)[1]
H_corr = freqz_parfilt(Filter_corr, worN=f, fs=fs, causal=False)[1]
H_dcm = freqz_parfilt(Filter_dcm, worN=f, fs=fs, causal=False)[1]
H_osdcm = freqz_parfilt(Filter_osdcm, worN=f, fs=10*fs, causal=False)[1]


# Plots
set_rcParams()
colors = cm.get_cmap('Greys_r')
fnorm = f / fNyquist
flim = fmin/fNyquist, fmax/fNyquist
mlim = -70, 10
elim = mlim
yfloor = -200
lw = 1.75
lw = 3.
kw_analog = dict(c='k', lw=0.75, ls='--', zorder=10, label='analog')
kw_fir = dict(c='#707070', ls=':', lw=lw, zorder=0)
kw_uncorr = dict(c='Black', ls=':', lw=lw, label='un-corrected')
kw_corr = dict(c='Black', ls='-', lw=lw, label='corrected')
kw_dcm = dict(c='#cccccc', ls=':', lw=lw, label='DC-matched')
kw_osdcm = dict(c='#cccccc', ls='-', lw=lw, alpha=1,
                label=r'DC (OS $\uparrow{}$)'.format(os))
kw_bl = dict(ls='-', lw=lw)
kw_labels = dict(fontsize=15)
kw_legend = dict(bbox_to_anchor=(1.05, 0, 0.58, 1), mode='expand',
                 borderaxespad=0, handlelength=1.5)
kw_savefig = dict(dpi=300, bbox_inches='tight')


# Conventional impulse invariant method (single-point correction)
fig, ax = plt.subplots(figsize=(7, 3), ncols=2, sharex=True, sharey=True,
                       gridspec_kw=dict(wspace=0.05))
handles = []
line, = ax[0].plot(fnorm, db(H_a), **kw_analog)
handles.append(line)
lines = [ax[0].plot(fnorm, db(H_uncorr), **kw_uncorr)[0],
         ax[0].plot(fnorm, db(H_corr), **kw_corr)[0],
         ax[0].plot(fnorm, db(H_dcm), **kw_dcm)[0],
         ax[0].plot(fnorm, db(H_osdcm), **kw_osdcm)[0]]
[handles.append(line) for line in lines]
ax[1].plot(fnorm, db(H_a - H_uncorr), **kw_uncorr)
ax[1].plot(fnorm, db(H_a - H_corr), **kw_corr)
ax[1].plot(fnorm, db(H_a - H_dcm), **kw_dcm)
ax[1].plot(fnorm, db(H_a - H_osdcm), **kw_osdcm)
for axi in ax:
    axi.set_xscale('log', base=2)
    axi.grid(True)
    axi.set_xticks(2.**np.array([-1, -4, -7]))
    axi.set_xlim(flim)
    axi.set_xlabel(r'Normalized frequency $\frac{\omega}{\omega_\textrm{s}/2}$',
                   **kw_labels)
    if axi.is_first_col():
        axi.set_ylim(mlim)
        axi.set_ylabel('Level in dB', **kw_labels)
        axi.set_title('Transfer Function')
    else:
        axi.set_title('Deviation')
legend = plt.legend(handles=handles, **kw_legend)
legend.get_frame().set_linewidth(0.5)

file_name = 'BPF-conventional-ii'
plt.savefig(file_name + '.pdf', **kw_savefig)
plt.savefig(file_name + '.png', **kw_savefig)


# Band-limited impulse invariant method
fig, ax = plt.subplots(figsize=(7, 3), ncols=2, sharex=True, sharey=True,
                       gridspec_kw=dict(wspace=0.05))
handles = []
line, = ax[0].plot(fnorm, db(H_a), **kw_analog)
handles.append(line)
for i, (Hi, (N, _)) in enumerate(zip(H_bl_stack, FIR_params)):
    coli = colors((i)/(len(FIR_params)))
    line, = ax[0].plot(fnorm, db(Hi), color=coli, label='$L={}$'.format(N),
                       **kw_bl)
    ax[1].plot(fnorm, db(H_a - Hi), color=coli, **kw_bl)
    handles.append(line)
line, = ax[0].plot(fnorm, db(H_fir_stack[-1]),
                   label='FIR ($L={}$)'.format(FIR_params[-1][0]), **kw_fir)
handles.append(line)
for axi in ax:
    axi.grid(True)
    axi.set_xscale('log', base=2)
    axi.set_xticks(2.**np.array([-1, -4, -7]))
    axi.set_xlim(flim)
    axi.set_xlabel(r'Normalized frequency $\frac{\omega}{\omega_\textrm{s}/2}$',
                   **kw_labels)
    if axi.is_first_col():
        axi.set_ylim(mlim)
        axi.set_ylabel('Level in dB', **kw_labels)
        axi.set_title('Transfer Function')
    else:
        axi.set_title('Deviation')
legend = ax[1].legend(handles=handles, **kw_legend)
legend.get_frame().set_linewidth(0.5)
axinset = ax[0].inset_axes([0.45, 0.15, 0.44, 0.44])
axinset.set_facecolor('white')
inset_flim = 16000*10**-0.03/fNyquist, 22000*10**0.03/fNyquist
idx = f > inset_flim[0]
mkinset = mark_inset(ax[0], axinset, loc1=2, loc2=4, fc='none', ec='k', lw=0.5)
axinset.plot(fnorm[idx], db(H_a[idx]), **kw_analog)
for i, (Hi, (N, _)) in enumerate(zip(H_bl_stack, FIR_params)):
    coli = colors((i)/len(FIR_params))
    axinset.plot(fnorm[idx], db(Hi[idx]), color=coli, **kw_bl)
axinset.set_xscale('log', base=2)
axinset.set_xlim(inset_flim)
axinset.set_ylim(-5.5, 0.5)
axinset.set_yticks([-5, 0], minor=False)
axinset.tick_params(which='both', labelsize=10)

file_name = 'BPF-blii'
plt.savefig(file_name + '.pdf', **kw_savefig)
plt.savefig(file_name + '.png', **kw_savefig)
