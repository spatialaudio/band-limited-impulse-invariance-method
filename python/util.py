import numpy as np
from matplotlib import rcParams
from scipy.signal import freqz
from scipy.special import exp1
from scipy.signal.filter_design import _cplxreal as cplxreal


def db(x, *, power=False):
    """
    Convert *x* to decibel.

    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If ``power=False`` (the default), *x* is squared before
        conversion.

    """
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def log_frequency(fmin, fmax, num_f, endpoint=True):
    """
    Logarithmic frequency axis.

    Parameters
    ----------
    fmin : float
        Lower limit.
    fmax : float
        Upper limit.
    num_f : int
        NUmber of frequency bins.
    endpoint : bool, optional
        If True, fmax is the last frequency. Default is True.

    Returns
    -------
    array_like
        Logarithmic frequency axis.

    """
    return np.logspace(np.log10(fmin), np.log10(fmax), num=num_f,
                       endpoint=endpoint)


def phaseshift_timedelay(delay, w):
    """
    Phase shift of a time delay

    Parameters
    ----------
    delay : float
        Delay in seconds.
    w : array_like
        Frequencies in Hertz.

    Returns
    -------
    array_like
        Phase shift.

    """
    return np.exp(-1j * 2 * np.pi * w * delay)


def phaseshift_sampledelay(n, w, fs):
    """
    Phase shift of a sample delay

    Parameters
    ----------
    n : float
        Sample delay. This can be also a non-integer.
    w : array_like
        Frequencies in Hertz.
    fs : int
        Sampling frequency in Hertz.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return phaseshift_timedelay(delay=n/fs, w=w)


def decaying_exponential(r, p, t, atzero=0.5):
    """
    Analytic impulse response of a first-order section filter.

    Parameters
    ----------
    r: complex
        Residue.
    p: complex
        Pole in the s-domain.
    t: array_like
        Time vector.
    atzero: float
        Value at the jump discontinuity, i.e. n=0. Default is 0.5.

    """
    h = np.zeros_like(t, dtype=complex)
    idx_zero = (t == 0)
    idx_pos = (t >= 0)
    h[idx_pos] = np.exp(p*t[idx_pos])
    h[idx_zero] = atzero
    return r * h


def decaying_sinusoid(r, p, t, atzero=0.5):
    """Analytic impulse response of a second-order section filter.

    Parameters
    ----------
    r: complex
        One from the complex conjugate residue. This must corresponds to the
        pole.
    p: complex
        One from the complex conjugate pole. This must corresponds to the
        residue.
    t: array_like
        Time vector.
    atzero: float
       Value at the jump discontinuity, i.e. n=0, Default is 0.5.
    """
    h = np.zeros_like(t)
    idx_zero = (t == 0)
    idx_pos = (t > 0)
    phi = np.angle(r)
    h[idx_pos] = \
        2 * np.abs(r) \
        * np.exp(p.real * t[idx_pos]) \
        * np.cos(-p.imag * t[idx_pos] + phi)
    h[idx_zero] = 2 * atzero * np.abs(r) * np.cos(phi)
    return h


def bandlimited_decaying_exponential(r, p, t, fs, residual=False, atzero=0.5):
    """
    Band-limited exponential (BLEX) function.

    This corresponds to the band-limited impulse response of a first-order
    section filter.

    Parameters
    ----------
    r: complex
        Complex residue.
    p: complex
        Complex pole.
    t: array_like
        Time vector.
    fs: int
        Sampling frequency in Hz.
    residual: bool
        If True, the BLEX residual is returned.
    atzero: float
       Value at the jump discontinuity, i.e. n=0, Default is 0.5.
    """

    h = np.zeros_like(t, dtype=complex)
    idx_zero = (t == 0)
    idx_pos = (t > 0)
    idx_neg = (t < 0)

    u1 = -1j*np.pi*fs + p  # lower limit
    u2 = 1j*np.pi*fs + p  # upper limit
    v1 = 1j*2*np.pi if residual else 0
    v2 = 0 if residual else 1j*2*np.pi

    h[idx_zero] = np.log(u1/u2) - atzero*v1
    h[idx_pos] = exp1(u2*t[idx_pos]) - exp1(u1*t[idx_pos]) + v2
    h[idx_neg] = exp1(u2*t[idx_neg]) - exp1(u1*t[idx_neg])
    return -1j/2/np.pi * r * h * np.exp(p*t)


def bandlimited_decaying_sinusoid(r, p, t, fs, residual=False, atzero=0.5):
    """
    Band-limited sinusoid function.

    This corresponds to the band-limited impulse response of a second-order
    section filter with conjugate complex poles.

    Parameters
    ----------
    r: array_like
        One from the complex conjugate residue. This must corresponds to the
        pole.
    p: array_like
        One from the complex conjugate pole. This must corresponds to the
        residue.
    t: array_like
        Time vector.
    fs: int
        Sampling frequency in Hz.
    residual: bool
        If True, the BLEX residual is returned.
    atzero: float
       Value at the jump discontinuity, i.e. n=0, Default is 0.5.

    """
    return 2 * bandlimited_decaying_exponential(r, p, t, fs, residual=residual,
                                                atzero=atzero).real


def impulse_invariance(r, p, k, L_fir, n_center, fs, mode, window=None):
    """
    Impulse invariance method.

    Parameters
    ----------
    r : array_like
        Residues.
    p : array_like
        Poles.
    k : float
        Direct throughput.
    L_fir : int
        FIR length.
    n_center : int
        Sample index at which the IIR fitler begins (pre-delay).
    fs : int
        Sampling frequency in Hertz.
    mode : string
        {"uncorrected", "corrected", "dcmatched", "bandlimited",
         "dcmbandlimited"}.
    window : array_like, optional
        Tapering window.

    Returns
    -------
    IIR : list
        List of FOS and SOS section filters.
    FIR : array_like
        FIR coefficients.
    n_center : int
        Sample index at which the IIR fitler begins (pre-delay).

    """
    p_cplx, p_real = cplxreal(p)
    num_real = len(p_real)
    r_cplx, r_real = r[num_real::2], r[:num_real]

    filters_real = [fos_real_pole(ri, pi, L_fir, n_center, fs, mode, window)
                    for (ri, pi) in zip(r_real, p_real)]
    filters_cplx = [sos_cplx_poles(ri, pi, L_fir, n_center, fs, mode, window)
                    for (ri, pi) in zip(r_cplx, p_cplx)]

    IIR = []
    FIR = np.zeros(L_fir)
    for filt in filters_real:
        IIR.append((filt[0], filt[1]))
        FIR += filt[2]
    for filt in filters_cplx:
        IIR.append((filt[0], filt[1]))
        FIR += filt[2]
    if len(k) == 1:
        FIR[n_center] += k
    return IIR, FIR, n_center


def fos_real_pole(r, p, L_fir, n_center, fs, mode, window=None):
    """
    Impulse invariance method applied to a first-order section filter
    with a real pole.

    Parameters
    ----------
    r : float
        residue.
    p : float
        pole.
    L_fir : int
        FIR length.
    n_center : int
        Sample index where the IIR fitler begins (pre-delay).
    fs : int
        Sampling frequency in Hertz.
    mode : string
        {"uncorrected", "corrected", "dcmatched", "bandlimited",
         "dcmbandlimited"}.
    window : array_like, optional
        Tapering window.

    Returns
    -------
    list
        Numerator coefficients.
    list
        Denominator coefficients.
    FIR : array_like
        FIR coefficients.

    """
    T = 1/fs
    rd = r.real*T
    pd = np.exp(p.real*T)

    # First-order section
    b0 = rd
    a1 = -pd

    if mode in {'bandlimited', 'dcmbandlimited'}:
        n = np.arange(L_fir) - n_center
        blexres = bandlimited_decaying_exponential(r, p, n/fs, fs,
                                                   residual=True)
        FIR = T * blexres.real
        if window is not None:
            FIR *= window
        if mode == 'dcmbandlimited':
            FIR[n_center] -= (r/p + rd/(1-pd)).real + np.sum(FIR)
    elif mode == 'corrected':
        FIR = np.zeros(L_fir)
        FIR[n_center] = -0.5 * rd
    elif mode == 'uncorrected':
        FIR = np.zeros(L_fir)
    elif mode == 'dcmatched':
        FIR = np.zeros(L_fir)
        FIR[n_center] = -(r/p + rd/(1-pd)).real
    else:
        FIR = np.zeros(L_fir)
    return [b0], [1, a1], FIR


def sos_cplx_poles(r, p, L_fir, n_center, fs, mode, window=None):
    """
    Impulse invariance method applied to a second-order section filter
    with complex conjugate poles.

    Parameters
    ----------
    r : complex
        One from the complex conjugate pole. This must corresponds to the
        pole given in the second argument.
    r: array_like
        One from the complex conjugate residue. This must corresponds to the
        residue given in the first argument.
    L_fir : int
        FIR length.
    n_center : int
        Sample index where the IIR fitler begins (pre-delay).
    fs : int
        Sampling frequency in Hertz.
    mode : string
        {"uncorrected", "corrected", "dcmatched", "bandlimited",
         "dcmbandlimited"}.
    window : array_like, optional
        Tapering window.

    Returns
    -------
    list
        Numerator coefficients.
    list
        Denominator coefficients.
    FIR : array_like
        FIR coefficients.

    """
    T = 1/fs
    rd = r*T
    pd = np.exp(p*T)

    # Second-order section
    b0 = 2 * r.real * T
    b1 = -2 * (r.conj() * pd).real * T
    a1 = -2 * np.exp(p.real*T) * np.cos(p.imag*T)
    a2 = np.exp(2 * p.real * T)

    if mode in {'bandlimited', 'dcmbandlimited'}:
        n = np.arange(L_fir) - n_center
        blexres = bandlimited_decaying_sinusoid(r, p, n/fs, fs, residual=True)
        FIR = T * blexres
        if window is not None:
            FIR *= window
        if mode == 'dcmbandlimited':
            FIR[n_center] -= 2 * (r/p + rd/(1-pd)).real + np.sum(FIR)
    elif mode == 'corrected':
        FIR = np.zeros(L_fir)
        FIR[n_center] = - rd.real
    elif mode == 'uncorrected':
        FIR = np.zeros(L_fir)
    elif mode == 'dcmatched':
        FIR = np.zeros(L_fir)
        FIR[n_center] = -2 * (r/p + rd/(1-pd)).real
    return [b0, b1], [1, a1, a2], FIR


def freqz_parfilt(ParallelFilter, worN=512, fs=1, causal=True):
    """
    Frequency response of a parallel connection of IIR and FIR filters.

    Parameters
    ----------
    ParallelFilter : list
        List of IIR and FIR coefficients.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies.
        If an array_like, compute the response at the frequencies given.
        These are in the same units as fs.
    fs : float, optional
        Sampling frequency in Hertz.
    causal : bool, optional
        If True, a pre-delay is applied.

    Returns
    -------
    w : ndarray
        The frequencies at which the transfer function is computed.
    H : ndarray
        Frequency response.

    """
    IIR, FIR, n0 = ParallelFilter
    w, H_fir = freqz(FIR, [1], worN=worN, fs=fs)
    H_iir = np.sum([freqz(*iir, worN=w, fs=fs)[1] for iir in IIR], axis=0)
    if causal is True:
        H_iir *= phaseshift_sampledelay(n0, w, fs)
    else:
        H_fir *= phaseshift_sampledelay(-n0, w, fs)
    return w, H_fir + H_iir


def set_rcParams():
    """
    Load parameters for Matplotlib.

    """
    rcParams['axes.linewidth'] = 0.5
    rcParams['axes.edgecolor'] = 'k'
    rcParams['axes.facecolor'] = 'None'
    rcParams['axes.labelcolor'] = 'black'
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = 'Times New Roman'
    rcParams['font.weight'] = 'normal'
    rcParams['font.size'] = 13
    rcParams['grid.linewidth'] = 0.25
    rcParams['legend.facecolor'] = 'white'
    rcParams['legend.fontsize'] = 13
    rcParams['legend.fancybox'] = False
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 1
    rcParams['legend.edgecolor'] = 'k'
    rcParams['savefig.bbox'] = 'tight'
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
