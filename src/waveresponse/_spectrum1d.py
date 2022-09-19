import numpy as np


class _PiersonMoskowitz:
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


class ModifiedPiersonMoskowitz(_PiersonMoskowitz):
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


class JONSWAP(ModifiedPiersonMoskowitz):
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
