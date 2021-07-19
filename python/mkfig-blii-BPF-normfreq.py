"""
Digital modeling of an analog prototype filter using impulse invariance method.

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
f_Nyquist = 0.5 * fs
N_os = 10  # oversampling rate

# frequencies
fmin, fmax, num_f = 93.8, 24000, 2000
f = log_frequency(fmin, fmax, num_f)
# w = 2*np.pi*f

# Peaking EQ (analog prototype filter)
fc = 0.5 * f_Nyquist
wc = 2*np.pi*fc  # center frequency
G = 15  # gain in dB
g = 10**(G/20)  # linear gain
Q = 1/np.sqrt(2)  # quality factor
b = np.array([0, 1/Q/wc, 0])  # numerator
a = np.array([1/wc**2, 1/Q/wc, 1])  # denominator
rpk = residue(b, a)  # partial fraction expansion (continuous-time)
# k = np.asarray([0])

# Digital filter design
FIR_params = [(5, 2), (11, 5), (21, 10), (41, 20)]
Filters_bl = [impulse_invariance(
    *rpk, *firpar, fs, 'dcmbandlimited', kaiser(firpar[0], beta=8.6))
    for firpar in FIR_params]
Filter_uncorr = impulse_invariance(*rpk, 1, 0, fs, 'uncorrected')
Filter_corr = impulse_invariance(*rpk, 1, 0, fs, 'corrected')
Filter_dcm = impulse_invariance(*rpk, 1, 0, fs, 'dcmatched')
Filter_osdcm = impulse_invariance(*rpk, 1, 0, N_os*fs, 'dcmatched')

# Frequency responses
H_a = freqs(b, a, worN=2*np.pi*f)[1]
H_bl_stack = [freqz_parfilt(Filter, f, fs, causal=False)[1]
              for Filter in Filters_bl]
H_fir_stack = [freqz(Filter[1], worN=f, fs=fs)[1] for Filter in Filters_bl]
H_uncorr = freqz_parfilt(Filter_uncorr, worN=f, fs=fs, causal=False)[1]
H_corr = freqz_parfilt(Filter_corr, worN=f, fs=fs, causal=False)[1]
H_dcm = freqz_parfilt(Filter_dcm, worN=f, fs=fs, causal=False)[1]
H_osdcm = freqz_parfilt(Filter_osdcm, worN=f, fs=N_os*fs, causal=False)[1]


# Plots
set_rcParams()
colors = cm.get_cmap('Greys_r')
w = f / f_Nyquist  # normalized frequency (1 corresponds to f_Nyquist)
wlim = fmin/f_Nyquist, fmax/f_Nyquist
mlim = -70, 10
elim = mlim
# yfloor = -200
lw = 3
kw_analog = dict(c='k', lw=0.75, ls='--', dashes=(5, 5), label='analog')
kw_uncorr = dict(c='k', ls='--', dashes=(1, 0.75), lw=lw, label='un-corrected')
kw_corr = dict(c='k', ls='-', lw=lw, label='corrected')
kw_dcm = dict(c='0.75', ls='--', dashes=(1, 0.75), lw=lw, label='DC-matched')
kw_osdcm = dict(c='0.75', ls='-', lw=lw,
                label=r'DC (OS $\uparrow{}$)'.format(N_os))
kw_bl = dict(ls='-', lw=lw)
kw_fir = dict(c='0.5', ls='--', dashes=(1, 0.75), lw=lw,
              label='FIR ($L={}$)'.format(FIR_params[-1][0]))
kw_subplots = dict(figsize=(7, 3), ncols=2, sharex=True, sharey=True,
                   gridspec_kw=dict(wspace=0.05))
kw_axislabels = dict(fontsize=15)
kw_legend = dict(bbox_to_anchor=(1.05, 0, 0.58, 1), mode='expand',
                 borderaxespad=0, handlelength=1.5)
kw_savefig = dict(dpi=300, bbox_inches='tight')

# Conventional impulse invariant method (single-point correction)
fig_name = 'BPF-conventional-ii-normfreq'
fig, ax = plt.subplots(**kw_subplots)
lines = [ax[0].plot(w, db(H_uncorr), **kw_uncorr),
         ax[0].plot(w, db(H_corr), **kw_corr),
         ax[0].plot(w, db(H_dcm), **kw_dcm),
         ax[0].plot(w, db(H_osdcm), **kw_osdcm)]
line_a = ax[0].plot(w, db(H_a), **kw_analog)
ax[1].plot(w, db(H_a - H_uncorr), **kw_uncorr)
ax[1].plot(w, db(H_a - H_corr), **kw_corr)
ax[1].plot(w, db(H_a - H_dcm), **kw_dcm)
ax[1].plot(w, db(H_a - H_osdcm), **kw_osdcm)
for axi in ax:
    axi.set_xscale('log', base=2)
    axi.grid(True)
    axi.set_xticks(2.**np.array([-1, -4, -7]))
    axi.set_xlim(wlim)
    axi.set_xlabel(
        r'Normalized frequency $\frac{\omega}{\omega_\textrm{s}/2}$',
        **kw_axislabels)
ax[0].set_ylim(mlim)
ax[0].set_ylabel('Level in dB', **kw_axislabels)
ax[0].set_title('Transfer Function')
ax[1].set_title('Deviation')
handles = []
handles.append(line_a[0])
[handles.append(line_i[0]) for line_i in lines]
legend = plt.legend(handles=handles, **kw_legend)
legend.get_frame().set_linewidth(0.5)
plt.savefig(fig_name + '.pdf', **kw_savefig)
plt.savefig(fig_name + '.png', **kw_savefig)

# Band-limited impulse invariant method
fig_name = 'BPF-blii-normfreq'
fig, ax = plt.subplots(**kw_subplots)
lines = []
for i, (Hi, (N, _)) in enumerate(zip(H_bl_stack, FIR_params)):
    coli = colors((i)/(len(FIR_params)))
    lines.append(
        ax[0].plot(w, db(Hi), color=coli, label='$L={}$'.format(N), **kw_bl))
    ax[1].plot(w, db(H_a - Hi), color=coli, **kw_bl)
line_fir = ax[0].plot(w, db(H_fir_stack[-1]), **kw_fir)
line_a = ax[0].plot(w, db(H_a), **kw_analog)
for axi in ax:
    axi.grid(True)
    axi.set_xscale('log', base=2)
    axi.set_xticks(2.**np.array([-1, -4, -7]))
    axi.set_xlim(wlim)
    axi.set_xlabel(
        r'Normalized frequency $\frac{\omega}{\omega_\textrm{s}/2}$',
        **kw_axislabels)
ax[0].set_ylim(mlim)
ax[0].set_ylabel('Level in dB', **kw_axislabels)
ax[0].set_title('Transfer Function')
ax[1].set_title('Deviation')
handles = []
handles.append(line_a[0])
[handles.append(line_i[0]) for line_i in lines]
handles.append(line_fir[0])
legend = ax[1].legend(handles=handles, **kw_legend)
legend.get_frame().set_linewidth(0.5)
axinset = ax[0].inset_axes([0.45, 0.15, 0.44, 0.44])
axinset.set_facecolor('white')
mark_inset(ax[0], axinset, loc1=2, loc2=4, fc='none', ec='k', lw=0.5)
inset_wlim = 14000*10**-0.03/f_Nyquist, 22000*10**0.03/f_Nyquist
idx = f > inset_wlim[0]
idx = np.logical_and(w > inset_wlim[0], w < inset_wlim[1])
for i, H_i in enumerate(H_bl_stack):
    coli = colors(i/len(FIR_params))
    axinset.plot(w[idx], db(H_i[idx]), color=coli, **kw_bl)
axinset.plot(w[idx], db(H_a[idx]), **kw_analog)
axinset.set_xscale('log', base=2)
axinset.set_xlim(inset_wlim)
axinset.set_ylim(-5.5, 0.5)
axinset.set_yticks([-5, 0], minor=False)
axinset.tick_params(which='both', labelsize=10)
plt.savefig(fig_name + '.pdf', **kw_savefig)
plt.savefig(fig_name + '.png', **kw_savefig)
