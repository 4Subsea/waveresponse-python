import numpy as np


def _robust_modulus(x, periodicity):
    """
    Robust modulus operator.

    Similar to ``x % periodicity``, but ensures that it is robust w.r.t. floating
    point numbers.
    """
    x = np.asarray_chkfinite(x % periodicity).copy()
    return np.nextafter(x, -1, where=(x == periodicity), out=x)


def complex_to_polar(complex_vals, phase_degrees=False):
    """
    Convert complex numbers to polar form (i.e., amplitude and phase).

    Parameters
    ----------
    complex_vals : array-like
        Complex number values.
    phase_degrees : bool
        Whether the phase angles should be returned in 'degrees' (``True``) or
        'radians' (``False``).

    Returns
    -------
    amp : array
        Amplitudes.
    phase : array
        Phase angles.

    """
    complex_vals = np.asarray_chkfinite(complex_vals)
    amp = np.abs(complex_vals)
    phase = np.angle(complex_vals, deg=phase_degrees)
    return amp, phase


def polar_to_complex(amp, phase, phase_degrees=False):
    """
    Convert polar coordinates (i.e., amplitude and phase) to complex numbers.

    Given as:

        ``A * exp(j * phi)``

    where, ``A`` is the amplitude and ``phi`` is the phase.

    Parameters
    ----------
    amp : array-like
        Amplitude values.
    phase : array-like
        Phase angle values.
    phase_degrees : bool
        Whether the phase angles are given in 'degrees'. If ``False``, 'radians'
        is assumed.

    Returns
    -------
    array :
        Complex numbers.
    """
    amp = np.asarray_chkfinite(amp)
    phase = np.asarray_chkfinite(phase)

    if phase_degrees:
        phase = (np.pi / 180.0) * phase

    if amp.shape != phase.shape:
        raise ValueError()

    return amp * np.exp(1j * phase)
