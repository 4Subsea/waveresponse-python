import numpy as np

from ._core import _robust_modulus, DirectionalSpectrum


class Spectrum:
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
