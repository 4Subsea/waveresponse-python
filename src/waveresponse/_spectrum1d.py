from abc import ABC, abstractstaticmethod

import numpy as np


class PiersonMoskowitz_:
    """
    Pierson-Moskowitz spectrum.

        ``S(w) = A/w**5 exp(-B/w**4)``
    """
    def __init__(self, freq, freq_hz=False):
        self._freq = np.asarray_chkfinite(freq).copy()
        self._freq_hz = freq_hz

        if self._freq_hz:
            self._freq *= 2.0 * np.pi

    def __call__(self, A, B, freq_hz=None):
        freq = self._freq.copy()
        spectrum = A / freq ** 5.0 * np.exp(-B / freq ** 4)

        if freq_hz is None:
            freq_hz = self._freq_hz

        if freq_hz:
            scale = (2.0 * np.pi)
        else:
            scale = 1.

        return freq / scale, spectrum * scale


class ModPiersonMoskowitz_(PiersonMoskowitz_):
    """
    Modified Pierson-Moskowitz (i.e., Bretschneider) spectrum.

        ``S(w) = A/w**5 exp(-B/w**4)``

    where,

        ``A = 5/16 * Hs**2 * w_p**4``,

        ``B = 5/4 * w_p**4``.
    """
    def __call__(self, hs, tp, freq_hz=None):
        omega_p = 2.0 * np.pi / tp

        A = (5.0 / 16.0) * hs ** 2.0 * omega_p ** 4.0
        B = (5.0 / 4.0) * omega_p ** 4

        return super().__call__(A, B, freq_hz=freq_hz)


class JONSWAP_(ModPiersonMoskowitz_):
    """
    JONSWAP spectrum.

        ``S(w) = C * S_pm(w) * gamma ** b

    where S_pm denotes the (modified) Pierson-Moskowitz spectrum,

        ``S_pm(w) = A/w**5 exp(-B/w**4)``

    and,

        ``A = 5/16 * Hs**2 * w_p**4``,

        ``B = 5/4 * w_p**4``,

        ``C = 1 - 0.287 * ln(gamma)``,

        ``b = exp(-(w - w_p)**2 / (2 * sigma**2 * wp**2))``,

        ``sigma = sigma_a``, for w <= wp
        ``sigma = sigma_b``, for w > wp
    """
    def __init__(self, freq, freq_hz=False, gamma=1, sigma_a=0.07, sigma_b=0.09):
        self._gamma = gamma
        self._sigma_a = sigma_a
        self._sigma_b = sigma_b
        super().__init__(freq, freq_hz=freq_hz)

    def __call__(self, hs, tp):
        gamma = self._gamma
        omega_p = 2.0 * np.pi / tp
        sigma = self._sigma(omega_p)
        C = 1.0 - 0.287 * np.log(gamma)
        b = np.exp(-0.5 * ((self._freq - omega_p) / (sigma - omega_p)) ** 2)

        freq, spectrum_pm = super().__call__(hs, tp)
        return freq, C * spectrum_pm * gamma ** b

    def _sigma(self, omega_p):
        """
        Spectrum parameter.
        """
        arg = self._freq <= omega_p
        sigma = np.empty_like(self._freq)
        sigma[arg] = self._sigma_a
        sigma[~arg] = self._sigma_b
        return sigma


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


class PiersonMoskowitz(BaseWave):
    """
    Pierson-Moskowitz wave spectrum.

    ``S(w) = A/w**5 exp(-B/w**4)``

    where ``A = 0.0081 * g**2`` and ``B = 1.25 * w_p**4``.

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.

    """

    @staticmethod
    def _spectrum(freq, A, B):
        """
        Base PiersonMaskowitz spectrum definition.
        """
        return A / freq ** 5.0 * np.exp(-B / freq ** 4)

    def _A(self, *args):
        """
        Spectrum parameter.
        """
        return 0.0081 * 9.80665 ** 2

    def _B(self, *args):
        """
        Spectrum parameter.
        """
        _, tp = args
        omega_p = 2.0 * np.pi / tp
        return 1.25 * omega_p ** 4


class ModPiersonMoskowitz(PiersonMoskowitz):
    """
    Modified Pierson-Moskowitz wave spectrum.

    ``S(w) = A/w**5 exp(-B/w**4)``

    where ``A = 5/16 * Hs**2 * w_p**4`` and ``B = 1.25 * w_p**4``.

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.

    """

    def _A(self, *args):
        """
        Spectrum parameter.
        """
        hs, tp = args
        omega_p = 2.0 * np.pi / tp

        A = 5.0 / 16.0 * hs ** 2.0 * omega_p ** 4.0
        return A


class JONSWAP(ModPiersonMoskowitz):
    """
    JONSWAP wave spectrum (derived from modified Pierson-Moskowitz).

    ``S(w) = A/w**5 exp(-B/w**4)``

    where ``A = 5/16 * Hs**2 * w_p**4 * (1 - 0.287*log(gamma)) * gamma**a``
    and ``B = 1.25 * w_p**4``. Furhtermore,

    ``a = exp(-(w - w_p)**2 / (2 * sigma**2 * wp**2))``,

    ``sigma = 0.07, for w <= wp`` and ``sigma = 0.09, for w > wp``.

    Note that ``gamma`` is user-defined.

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.
    gamma : float or callable
        A scalar value or a callable that accepts hs and tp, and returns
        a scalar value.

    """

    def __init__(self, freq, freq_hz=False, gamma=1.0):
        if not callable(gamma):
            self._gamma = lambda *args: gamma
        else:
            self._gamma = gamma
        super().__init__(freq, freq_hz=freq_hz)

    def _A(self, *args):
        """
        Spectrum parameter.
        """
        gamma = self._gamma(*args)
        a = self._a(*args)

        A = super()._A(*args)
        k = (1.0 - 0.287 * np.log(gamma)) * gamma ** a
        return A * k

    def _a(self, *args):
        """
        Spectrum parameter.
        """
        _, tp = args
        omega_p = 2.0 * np.pi / tp
        sigma = self._sigma(*args)

        a = np.exp(-(((self._freq - omega_p) / (sigma * omega_p)) ** 2.0) / 2.0)
        return a

    def _sigma(self, *args):
        """
        Spectrum parameter.
        """
        _, tp = args
        omega_p = 2.0 * np.pi / tp

        arg = self._freq <= omega_p
        sigma = np.empty_like(self._freq)
        sigma[arg] = 0.07
        sigma[~arg] = 0.09
        return sigma


class OchiHubble(ModPiersonMoskowitz):
    """
    Ochi-Hubble wave spectrum (derived from modified Pierson-Moskowitz).

    ``S(w) = A/w**5 exp(-B/w**4)``

    where ``A = ((4 * q + 1) / 4 * w_p**4)**q * Hs**2 / (4 * gamma(q) * w**(4 * (q - 1)))
    and ``B = (4 * q + 1) / 4 * w_p**4``.

    where ``q`` is a user-defined shape parameter. Note that ``gamma`` is the
    "Gamma function".

    Notes
    -----
    The special case ``q=1`` corresponds to the Modified Pierson-Moskowitz
    spectrum.

    Parameters
    ----------
    freq : array-like
        Sequence of frequencies to use when generating the spectrum.
    freq_hz : bool
        Whether the provided frequencies are in rad/s (default) or Hz.
    q : float or callable
        A scalar value or a callable that accepts hs and tp, and returns
        a scalar value.

    """

    def __init__(self, freq, freq_hz=False, q=1.0):
        if not callable(q):
            self._q = lambda *args: q
        else:
            self._q = q
        super().__init__(freq, freq_hz=freq_hz)

    def _A(self, *args):
        """
        Spectrum parameter.
        """
        q = self._q(*args)

        hs, tp = args
        omega_p = 2.0 * np.pi / tp
        a = ((4.0 * q + 1.0) / 4.0 * omega_p ** 4) ** q
        A = (a * hs ** 2) / (4 * gammafun(q) * self._freq ** (4 * (q - 1.0)))
        return A

    def _B(self, *args):
        """
        Spectrum parameter.
        """
        q = self._q(*args)

        _, tp = args
        omega_p = 2.0 * np.pi / tp
        return (4 * q + 1) / 4 * omega_p ** 4
