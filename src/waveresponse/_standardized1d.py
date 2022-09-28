from abc import ABC, abstractmethod

import numpy as np
from scipy.special import gamma as gammafun


class BaseSpectrum1d(ABC):
    """
    Base class for handling and generating 1-D wave spectra.

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

    def __call__(self, *args, freq_hz=None, **kwargs):
        """
        Generate wave spectrum.

        Parameters
        ----------
        *args
            Spectrum parameters.
        freq_hz : bool, optional
            Whether to return the frequencies and spectrum in terms of Hz (`True`)
            or rad/s (`False`). If `None` (default), the original units of `freq` is
            preserved.
        **kwargs
            Spectrum parameters.

        Return
        ------
        freq : 1-D array
            Frequencies corresponding to the spectrum values. Unit is set according
            to `freq_hz`.
        spectrum : 1-D array
            Spectrum values. Unit is set according to `freq_hz`.

        Notes
        -----
        The scaling between wave spectrum in terms of Hz and rad/s is defined
        as:

        ``S(f) = 2*pi*S(w)``

        where ``S(f)`` and ``S(w)`` are the same spectrum but expressed
        in terms of Hz and rad/s, respectively.
        """
        freq = self._freq.copy()
        spectrum = self._spectrum(freq, *args, **kwargs)

        if freq_hz is None:
            freq_hz = self._freq_hz

        if freq_hz:
            scale = 2.0 * np.pi
        else:
            scale = 1.0

        return freq / scale, spectrum * scale

    @abstractmethod
    def _spectrum(self, omega, *args, **kwargs):
        raise NotImplementedError()


class BasePMSpectrum(BaseSpectrum1d):
    """
    Base class for handling 1-D Pierson-Moskowtiz (PM) type spectra of the form:

    ``S(w) = A / w ** 5 exp(-B / w ** 4)``

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.

    See Also
    --------
    ModifiedPiersonMoskowitz : Modified Pierson-Moskowitz wave spectrum.
    JONSWAP : JONSWAP wave spectrum.
    OchiHubble : Ochi-Hubble (three-parameter) wave spectrum.
    """

    def __call__(self, A, B, freq_hz=None):
        """
        Generate wave spectrum.

        Parameters
        ----------
        A : float
            Spektrum shape parameter.
        B : float
            Spektrum shape parameter.
        freq_hz : bool, optional
            Whether to return the frequencies and spectrum in terms of Hz (`True`)
            or rad/s (`False`). If `None` (default), the original units of `freq` is
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
        The scaling between wave spectrum in terms of Hz and rad/s is defined
        as:

        ``S(f) = 2*pi*S(w)``

        where ``S(f)`` and ``S(w)`` are the same spectrum but expressed
        in terms of Hz and rad/s, respectively.
        """
        return super().__call__(A, B, freq_hz=freq_hz)

    def _spectrum(self, omega, A, B):
        return A / omega**5.0 * np.exp(-B / omega**4)


class ModifiedPiersonMoskowitz(BasePMSpectrum):
    """
    Modified Pierson-Moskowitz (i.e., Bretschneider) spectrum, given by:

    ``S(w) = A / w ** 5 exp(-B / w ** 4)``

    where,

    - ``A = 5 / 16 * Hs ** 2 * w_p ** 4``
    - ``B = 5 / 4 * w_p ** 4``.
    - ``Hs`` is the significant wave height.
    - ``w_p = 2pi / Tp`` is the angular spectral peak frequency.

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.

    See Also
    --------
    JONSWAP : JONSWAP wave spectrum.
    OchiHubble : Ochi-Hubble (three-parameter) wave spectrum.
    """

    def __call__(self, hs, tp, freq_hz=None):
        """
        Generate wave spectrum.

        Parameters
        ----------
        hs : float
            Significant wave height, Hs.
        tp : float
            Peak period, Tp.
        freq_hz : bool, optional
            Whether to return the frequencies and spectrum in terms of Hz (`True`)
            or rad/s (`False`). If `None` (default), the original units of `freq` is
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
        The scaling between wave spectrum in terms of Hz and rad/s is defined
        as:

        ``S(f) = 2*pi*S(w)``

        where ``S(f)`` and ``S(w)`` are the same spectrum but expressed
        in terms of Hz and rad/s, respectively.
        """

        A = self._A(hs, tp)
        B = self._B(tp)

        return super().__call__(A, B, freq_hz=freq_hz)

    def _A(self, hs, tp):
        omega_p = 2.0 * np.pi / tp
        return (5.0 / 16.0) * hs**2.0 * omega_p**4.0

    def _B(self, tp):
        omega_p = 2.0 * np.pi / tp
        return (5.0 / 4.0) * omega_p**4


class JONSWAP(ModifiedPiersonMoskowitz):
    """
    JONSWAP spectrum, given as:

    ``S(w) = alpha * S_pm(w) * gamma ** b``

    where,

    ``b = exp(-(w - w_p) ** 2 / (2 * sigma ** 2 * wp ** 2))``

    and,

    - ``S_pm(w)`` is the Pierson-Moskowitz (PM) spectrum.
    - ``gamma`` is a peak enhancement factor.
    - ``alpha = 1 - 0.287 * ln(gamma)`` is a normalizing factor.
    - ``sigma`` is the spectral width parameter (established from experimental data):
        - ``sigma = 0.07`` for ``w <= wp``
        - ``sigma = 0.09`` for ``w > wp``
    - ``wp = 2pi / Tp`` is the angular spectral peak frequency.

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.

    Notes
    -----
    The special case ``gamma=1`` corresponds to the modified Pierson-Moskowitz spectrum.

    See Also
    --------
    ModifiedPiersonMoskowitz : Modified Pierson-Moskowitz wave spectrum.
    OchiHubble : Ochi-Hubble (three-parameter) wave spectrum.
    """

    def __call__(self, hs, tp, gamma=1, freq_hz=None):
        """
        Generate wave spectrum.

        Parameters
        ----------
        hs : float
            Significant wave height, Hs.
        tp : float
            Peak period, Tp.
        gamma : float
            Peak enhancement factor.
        freq_hz : bool, optional
            Whether to return the frequencies and spectrum in terms of Hz (`True`)
            or rad/s (`False`). If `None` (default), the original units of `freq` is
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
        The scaling between wave spectrum in terms of Hz and rad/s is defined
        as:

        ``S(f) = 2*pi*S(w)``

        where ``S(f)`` and ``S(w)`` are the same spectrum but expressed
        in terms of Hz and rad/s, respectively.
        """
        alpha = self._alpha(gamma)
        b = self._b(tp)

        freq, spectrum_pm = super().__call__(hs, tp, freq_hz=freq_hz)

        return freq, alpha * spectrum_pm * gamma**b

    def _alpha(self, gamma):
        return 1.0 - 0.287 * np.log(gamma)

    def _b(self, tp):
        omega_p = 2.0 * np.pi / tp
        sigma = self._sigma(omega_p)
        return np.exp(-0.5 * ((self._freq - omega_p) / (sigma * omega_p)) ** 2)

    def _sigma(self, omega_p):
        """
        Spectral width.
        """
        arg = self._freq <= omega_p
        sigma = np.empty_like(self._freq)
        sigma[arg] = 0.07
        sigma[~arg] = 0.09
        return sigma


class OchiHubble(BaseSpectrum1d):
    """
    Ochi-Hubble wave spectrum, given as:

    ``S(w) = C / w ** (4 * q + 1) * exp(-d / w ** 4)``

    where,

    - ``C = (1.0 / 4.0) * (c ** q * Hs ** 2) / gamma(q)``
    - ``c = (4.0 * q + 1.0) * w_p ** 4 / 4.0``
    - ``d = (4 * q + 1) * w_p ** 4 / 4.0``
    - ``Hs`` is the significant wave height.
    - ``w_p = 2pi / Tp`` is the angular spectral peak frequency.

    and ``q`` is a user-defined shape parameter. Note that ``gamma`` is the "Gamma function".

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.

    Notes
    -----
    The special case ``q=1`` corresponds to the modified Pierson-Moskowitz spectrum.

    See Also
    --------
    ModifiedPiersonMoskowitz : Modified Pierson-Moskowitz wave spectrum.
    JONSWAP : JONSWAP wave spectrum.
    """

    def __call__(self, hs, tp, q=2, freq_hz=None):
        """
        Generate wave spectrum.

        Parameters
        ----------
        hs : float
            Significant wave height, Hs.
        tp : float
            Peak period, Tp.
        q : float
            Spectral shape parameter.
        freq_hz : bool, optional
            Whether to return the frequencies and spectrum in terms of Hz (`True`)
            or rad/s (`False`). If `None` (default), the original units of `freq` is
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
        The scaling between wave spectrum in terms of Hz and rad/s is defined
        as:

        ``S(f) = 2*pi*S(w)``

        where ``S(f)`` and ``S(w)`` are the same spectrum but expressed
        in terms of Hz and rad/s, respectively.
        """

        return super().__call__(hs, tp, q=q, freq_hz=freq_hz)

    def _spectrum(self, omega, hs, tp, q):
        C = self._C(hs, tp, q)
        d = self._d(tp, q)

        return C / omega ** (4 * q + 1) * np.exp(-d / omega**4)

    def _C(self, hs, tp, q):
        omega_p = 2.0 * np.pi / tp
        c = (4.0 * q + 1.0) * omega_p**4 / 4.0
        return (1.0 / 4.0) * (c**q * hs**2) / gammafun(q)

    def _d(self, tp, q):
        omega_p = 2.0 * np.pi / tp
        return (4 * q + 1) * omega_p**4 / 4.0
