import copy

import numpy as np
from scipy.interpolate import interp2d


class Grid:
    """
    Frequency / direction grid.

    Parameters
    ----------
    freq : array-like
        Frequency bins. Positive and monotonically increasing.
    dirs : array-like
        Direction bins. Positive and monotonically increasing. Must cover the directional
        range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
    vals : array-like (N, M)
        Value grid as 2-D array of shape (N, M), such that
        ``N=len(freq)`` and ``M=len(dirs)``.
    freq_hz : bool
        If frequency is given in Hz. If False, rad/s is assumed.
    degrees : bool
        If direction is given in degrees. If False, radians are assumed.
    clockwise : bool
        If positive directions are defined to be 'clockwise'. If False, 'counterclockwise'
        is assumed.
    waves_coming_from : bool
        If waves are 'coming from' the given directions. If False, 'going towards'
        convention is assumed.
    """

    def __init__(
        self,
        freq,
        dirs,
        vals,
        freq_hz=False,
        degrees=False,
        clockwise=True,
        waves_coming_from=True,
    ):
        self._freq = np.asarray_chkfinite(freq).copy()
        self._dirs = np.asarray_chkfinite(dirs).copy()
        self._vals = np.asarray_chkfinite(vals).copy()
        self._clockwise = clockwise
        self._waves_coming_from = waves_coming_from

        if freq_hz:
            self._freq = 2.0 * np.pi * self._freq

        if degrees:
            self._dirs = (np.pi / 180.0) * self._dirs

        self._check_freq(self._freq)
        self._check_dirs(self._dirs)
        if self._vals.shape != (len(self._freq), len(self._dirs)):
            raise ValueError(
                "Values must have shape shape (N, M), such that ``N=len(freq)`` "
                "and ``M=len(dirs)``."
            )

    def _check_freq(self, freq):
        """
        Check frequency bins.
        """
        if np.any(freq[:-1] >= freq[1:]) or freq[0] < 0:
            raise ValueError("Frequencies must be positive monotonically increasing.")

    def _check_dirs(self, dirs):
        """
        Check direction bins.
        """
        if np.any(dirs[:-1] >= dirs[1:]) or dirs[0] < 0 or dirs[-1] >= 2.0 * np.pi:
            raise ValueError(
                "Directions must be positive monotonically increasing and "
                "be [0., 360.) degs (or [0., 2*pi) rads)."
            )

    @property
    def wave_convention(self):
        """
        Wave convention.
        """
        return {
            "clockwise": self._clockwise,
            "waves_coming_from": self._waves_coming_from,
        }

    def set_wave_convention(self, clockwise=True, waves_coming_from=True):
        """
        Set wave convention.

        Directions and values will be converted (in-place) to the new convention.

        Parameters:
        -----------
        clockwise : bool
            If positive directions are defined to be 'clockwise'. If False, 'counterclockwise'
            is assumed.
        waves_coming_from : bool
            If waves are 'coming from' the given directions. If False, 'going towards'
            convention is assumed.
        """
        conv_org = self.wave_convention
        conv_new = {"clockwise": clockwise, "waves_coming_from": waves_coming_from}
        self._freq, self._dirs, self._vals = self._convert(
            self._freq, self._dirs, self._vals, conv_new, conv_org
        )
        self._clockwise = conv_new["clockwise"]
        self._waves_coming_from = conv_new["waves_coming_from"]

    def _convert(self, freq, dirs, vals, config_new, config_org):
        """
        Convert grid from one wave convention to another.
        """
        freq_org = np.asarray_chkfinite(freq).copy()
        dirs_org = np.asarray_chkfinite(dirs).copy()
        vals_org = np.asarray_chkfinite(vals).copy()

        freq_new = freq_org
        dirs_new = self._convert_dirs(dirs_org, config_new, config_org, degrees=False)
        dirs_new, vals_new = self._sort(dirs_new, vals_org)

        return freq_new, dirs_new, vals_new

    @staticmethod
    def _convert_dirs(dirs, config_new, config_org, degrees=False):
        """
        Convert wave directions from one convention to another.

        Parameters
        ----------
        dirs : float or array-like
            Wave directions in radians expressed according to 'original' convention.
        config_new : dict
            New wave direction convention.
        config_org : dict
            Original wave direction convention.

        Return
        ------
        dirs : numpy.array
            Wave directions in radians expressed according to 'new' convention.
        """
        dirs = np.asarray_chkfinite(dirs).copy()

        if degrees:
            periodicity = 360.0
        else:
            periodicity = 2.0 * np.pi

        if config_new["waves_coming_from"] != config_org["waves_coming_from"]:
            dirs -= periodicity / 2
        if config_new["clockwise"] != config_org["clockwise"]:
            dirs *= -1

        return dirs % periodicity

    @staticmethod
    def _sort(dirs, vals):
        """
        Sort directions and values according to (unsorted) directions.
        """
        dirs = np.asarray_chkfinite(dirs)
        vals = np.asarray_chkfinite(vals)
        sorted_args = np.argsort(dirs)
        return dirs[sorted_args], vals[:, sorted_args]
