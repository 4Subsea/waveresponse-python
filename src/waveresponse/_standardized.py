from abc import ABC, abstractmethod

import numpy as np


class BasePMSpectrum(ABC):
    """
    Base class for handling 1-D Pierson-Moskowtiz (PM) type spectra of the form:

    ``S(w) = A/w**5 exp(-B/w**4)``

    This class requires that the spectrum parameters, ``A`` and ``B``, can be calculated
    from the significant wave height, Hs, and the wave peak period, Tp.

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
        freq = self._freq.copy()
        spectrum = self._spectrum(freq, hs, tp)

        if freq_hz is None:
            freq_hz = self._freq_hz

        if freq_hz:
            scale = 2.0 * np.pi
        else:
            scale = 1.0

        return freq / scale, spectrum * scale

    def _spectrum(self, omega, hs, tp):
        A = self._A(hs, tp)
        B = self._B(hs, tp)
        return A / omega**5.0 * np.exp(-B / omega**4)

    @abstractmethod
    def _A(self, hs, tp):
        raise NotImplementedError()

    @abstractmethod
    def _B(self, hs, tp):
        raise NotImplementedError()


class ModifiedPiersonMoskowitz(BasePMSpectrum):
    """
    Modified Pierson-Moskowitz (i.e., Bretschneider) spectrum, given by:

    ``S(w) = A/w**5 exp(-B/w**4)``

    where ``A = 5/16 * Hs**2 * w_p**4`` and ``B = 5/4 * w_p**4``. ``Hs`` is the
    significant wave height, and ``w_p = 2pi / Tp`` is the peak frequency of the
    spectrum.

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

    def _A(self, *args):
        hs, tp = args
        omega_p = 2.0 * np.pi / tp
        return (5.0 / 16.0) * hs**2.0 * omega_p**4.0

    def _B(self, *args):
        _, tp = args
        omega_p = 2.0 * np.pi / tp
        return (5.0 / 4.0) * omega_p**4


class JONSWAP(ModifiedPiersonMoskowitz):
    """
    JONSWAP spectrum, given as:

    ``S(w) = alpha * S_pm(w) * gamma ** b``

    where,

    ``b = exp(-(w - w_p)**2 / (2 * sigma**2 * wp**2))``

    and,

    - ``S_pm(w)`` is the Pierson-Moskowitz (PM) spectrum.
    - ``gamma`` is a peak enhancement factor.
    - ``alpha = 1 - 0.287 * ln(gamma)`` is a normalizing factor.
    - ``sigma`` is the spectral width parameter:
        - ``sigma = simga_a`` for ``w <= wp``
        - ``sigma = sigma_b`` for ``w > wp``
    - ``wp = 2pi/tp`` spectral peak frequency.

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.
    gamma : float or callable
        Peak enhancement factor. Single value or a callable that accepts hs and tp,
        and returns a corresponding enhancement factor (float).
    sigma_a : float
        Spectral width parameter.
    sigma_b : float
        Spectral width parameter.

    See Also
    --------
    ModifiedPiersonMoskowitz : Pierson-Moskowitz (PM) wave spectrum.
    """

    def __init__(self, freq, freq_hz=False, gamma=1, sigma_a=0.07, sigma_b=0.09):
        if not callable(gamma):
            self._gamma = lambda *args: gamma
        else:
            self._gamma = gamma
        self._sigma_a = sigma_a
        self._sigma_b = sigma_b
        super().__init__(freq, freq_hz=freq_hz)

    def _spectrum(self, omega, hs, tp):
        gamma = self._gamma(hs, tp)
        alpha = 1.0 - 0.287 * np.log(gamma)
        omega_p = 2.0 * np.pi / tp
        sigma = self._sigma(omega_p)
        b = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)

        return alpha * super()._spectrum(omega, hs, tp) * gamma**b

    def _sigma(self, omega_p):
        """
        Spectral width.
        """
        arg = self._freq <= omega_p
        sigma = np.empty_like(self._freq)
        sigma[arg] = self._sigma_a
        sigma[~arg] = self._sigma_b
        return sigma
