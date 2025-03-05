import numpy as np
from scipy.integrate import trapezoid

from ._core import _robust_modulus, DirectionalSpectrum


class Spectrum1d:
    def __init__(self, freq, vals, freq_hz=True, clockwise=False, waves_coming_from=True):
        self._freq = np.asarray_chkfinite(freq).copy()
        self._vals = np.asarray_chkfinite(vals).copy()
        self._freq_hz = freq_hz
        self._clockwise = clockwise
        self._waves_coming_from = waves_coming_from

        if freq_hz:
            self._freq = 2.0 * np.pi * self._freq

    def freq(self, freq_hz=None):
        """
        Frequency coordinates.

        Parameters
        ----------
        freq_hz : bool
            If frequencies should be returned in 'Hz'. If ``False``, 'rad/s' is used.
            Defaults to original units used during initialization.
        """
        freq = self._freq.copy()

        if freq_hz is None:
            freq_hz = self._freq_hz

        if freq_hz:
            freq = 1.0 / (2.0 * np.pi) * freq

        return freq
    
    def as_directional(self, dirs, spread_fun, dirp, degrees=False):
        vals = self._vals.reshape(-1, 1)
        freq = self._freq.copy()

        vals = np.tile(vals, (1, len(dirs)))

        if degrees:
            period = 360.0
        else:
            period = 2.0 * np.pi

        if self._freq_hz:
            freq = freq / (2.0 * np.pi) 

        for (idx_f, idx_d), val_i in np.ndenumerate(vals):
            f_i = freq[idx_f]
            d_i = _robust_modulus(dirs[idx_d] - dirp, period)
            vals[idx_f, idx_d] = spread_fun(f_i, d_i) * val_i

        return DirectionalSpectrum(
            freq,
            dirs,
            vals,
            freq_hz=self._freq_hz,
            degrees=degrees,
            clockwise=self._clockwise,
            waves_coming_from=self._waves_coming_from,
        )

    def moment(self, n, freq_hz=None):
        """
        Calculate spectral moment (along the frequency domain).

        Parameters
        ----------
        n : int
            Order of the spectral moment.
        freq_hz : bool
            If frequencies in 'Hz' should be used. If ``False``, 'rad/s' is used.
            Defaults to original unit used during initialization.

        Returns
        -------
        float :
            Spectral moment.

        Notes
        -----
        The spectral moment is calculated according to Equation (8.31) and (8.32)
        in reference [1].

        References
        ----------
        [1] A. Naess and T. Moan, (2013), "Stochastic dynamics of marine structures",
        Cambridge University Press.

        """

        freq = self._freq.copy()
        vals = self._vals.copy()

        if freq_hz is None:
            freq_hz = self._freq_hz

        if freq_hz:
            vals *= 2.0 * np.pi

        m_n = trapezoid((freq**n) * vals, self._freq)
        return m_n
    

class WaveSpectrum1d(Spectrum1d):

    @property
    def hs(self):
        """
        Significan wave height, Hs.

        Calculated from the zeroth-order spectral moment according to:

        ``hs = 4.0 * sqrt(m0)``

        Notes
        -----
        The significant wave height is calculated according to equation (2.26) in
        reference [1].

        References
        ----------
        [1] 0. M. Faltinsen, (1990), "Sea loads on ships and offshore structures",
        Cambridge University Press.
        """
        m0 = self.moment(0)
        return 4.0 * np.sqrt(m0)
