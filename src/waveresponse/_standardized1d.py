import numpy as np


class BasePMSpectrum:
    """
    Base class for handling 1-D Pierson-Moskowtiz (PM) type spectra of the form:

    ``S(w) = A/w**5 exp(-B/w**4)``

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.
    """

    def __init__(self, freq, freq_hz=False):
        self._freq = np.asarray_chkfinite(freq).copy()
        self._freq_hz = freq_hz

        if self._freq_hz:
            self._freq *= 2.0 * np.pi

    def __call__(self, A, B, freq_hz=None):
        """
        Generate wave spectrum given Hs and Tp.

        Parameters
        ----------
        hs : float
            Significant wave height, Hs.
        tp : float
            Peak period, Tp.
        freq_hz : bool, optional
            Whether to return the frequencies and spectrum in terms of rad/s (`True`)
            or Hz (`False`). If `None` (default), the original units of `freq` is
            preserved.

        Return
        ------
        freq : 1-D array
            Frequencies corresponding to the spectrum values. Unit is set according
            to `freq_hz`.
        spectrum : 1-D array
            Spectrum values. Unit is set according to `freq_hz`.

        Notes
        -----
        The scaling between wave spectrum in terms of Hz and rad/s is defind
        as:

        ``S(f) = 2*pi*S(w)``

        where ``S(f)`` and ``S(w)`` are the same spectrum but expressed
        in terms of Hz and rad/s, respectively.
        """
        freq = self._freq.copy()
        spectrum = self._spectrum(freq, A, B)

        if freq_hz is None:
            freq_hz = self._freq_hz

        if freq_hz:
            scale = 2.0 * np.pi
        else:
            scale = 1.0

        return freq / scale, spectrum * scale

    def _spectrum(self, omega, A, B):
        return A / omega**5.0 * np.exp(-B / omega**4)


class ModifiedPiersonMoskowitz(BasePMSpectrum):
    """
    Modified Pierson-Moskowitz (i.e., Bretschneider) spectrum, given by:

    ``S(w) = A/w**5 exp(-B/w**4)``

    where ``A = 5/16 * Hs**2 * w_p**4`` and ``B = 5/4 * w_p**4``. ``Hs`` is the
    significant wave height, and ``w_p = 2pi / Tp`` is the angular spectral peak
    frequency.

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.

    See Also
    --------
    JONSWAP : JONSWAP wave spectrum.
    """

    def __call__(self, hs, tp, freq_hz=None):
        """
        Generate wave spectrum given Hs and Tp.

        Parameters
        ----------
        hs : float
            Significant wave height, Hs.
        tp : float
            Peak period, Tp.
        freq_hz : bool, optional
            Whether to return the frequencies and spectrum in terms of rad/s (`True`)
            or Hz (`False`). If `None` (default), the original units of `freq` is
            preserved.

        Return
        ------
        freq : 1-D array
            Frequencies corresponding to the spectrum values. Unit is set according
            to `freq_hz`.
        spectrum : 1-D array
            Spectrum values. Unit is set according to `freq_hz`.

        Notes
        -----
        The scaling between wave spectrum in terms of Hz and rad/s is defind
        as:

        ``S(f) = 2*pi*S(w)``

        where ``S(f)`` and ``S(w)`` are the same spectrum but expressed
        in terms of Hz and rad/s, respectively.
        """

        A = self._A(hs, tp)
        B = self._B(hs, tp)

        return super().__call__(A, B)

    def _A(self, *args):
        hs, tp = args
        omega_p = 2.0 * np.pi / tp
        return (5.0 / 16.0) * hs**2.0 * omega_p**4.0

    def _B(self, *args):
        _, tp = args
        omega_p = 2.0 * np.pi / tp
        return (5.0 / 4.0) * omega_p**4
