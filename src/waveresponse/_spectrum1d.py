from abc import ABC, abstractstaticmethod

import numpy as np


class BaseWave(ABC):
    """
    Base class for standardized 1-D wave spectra.

    ``S(w) = A/w**5 exp(-B/w**4)``

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.
    """

    def __init__(self, freq, freq_hz=False):
        self._freq_hz = freq_hz
        self._freq = np.asarray_chkfinite(freq).copy()

        if self._freq_hz:
            self._freq *= 2.0 * np.pi

    def __call__(self, hs, tp, freq_hz=None):
        """
        Generate wave spectrum based on given Hs and Tp.

        Parameters
        ----------
        hs : float
            Significant wave height in meters.
        tp : float
            Peak wave period in seconds.
        freq_hz : bool, optional
            Whether to return the frequencies and spectrum in terms of rad/s (``True``)
            or Hz (``False``). If ``None`` (default), the original units of `freq` is
            preserved.

        Return
        ------
        freq : 1D array
            Frequencies corresponding to the spectrum values. Unit is set
            according to `freq_hz`.
        spectrum : 1D array
            Spectrum values. Unit is set according to `freq_hz`.

        Notes
        -----
        The scaling between wave spectrum in terms of Hz and rad/s is defind
        as:

        ``S(f) = 2*pi*S(w)``

        where ``S(f)`` and ``S(w)`` are the same spectrum but expressed
        in terms of Hz and rad/s, respectively.

        """
        args = (hs, tp)
        A = self._A(*args)
        B = self._B(*args)

        if freq_hz is None:
            freq_hz = self._freq_hz

        if freq_hz:
            scale = (2.0 * np.pi)
        else:
            scale = 1.

        return self._freq / scale, self._spectrum(self._freq, A, B) * scale

    @abstractstaticmethod
    def _spectrum(freq, A, B):
        """
        Override to specify spectrum definition in terms of rad/s.
        """
        raise NotImplementedError()
