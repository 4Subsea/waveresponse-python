import copy

import numpy as np
from scipy.interpolate import interp2d


def complex_to_polar(complex_vals, phase_degrees=False):
    """
    Convert complex numbers to polar form (i.e., amplitude and phase).

    Parameters
    ----------
    complex_vals : array-like
        Complex number values.
    phase_degrees : bool
        Weather the phase angle should be returned in degrees. If ``False``, radians
        is assumed.

    Returns
    -------
    amp : array
        Amplitudes.
    phase : array
        Phase angles.
    """
    complex_vals = np.asarray_chkfinite(complex_vals).copy()
    amp = np.abs(complex_vals)
    phase = np.angle(complex_vals, deg=phase_degrees)
    return amp, phase


def polar_to_complex(amp, phase, phase_degrees=False):
    amp = np.asarray_chkfinite(amp).copy()
    phase = np.asarray_chkfinite(phase).copy()

    if phase_degrees:
        phase = (np.pi / 180.0) * phase

    if amp.shape != phase.shape:
        raise ValueError()

    return amp * (np.cos(phase) + 1j * np.sin(phase))


# class GridAritmeticsMixin:
#     def __mul__(self, other):
#         if not isinstance(other, Grid):
#             raise ValueError()
#         elif np.any(self._freq != other._freq) or np.any(self._dirs != other._dirs):
#             raise ValueError()
#         elif self.wave_convention != other.wave_convention:
#             raise ValueError()

#         if isinstance(self, DirectionalSpectrum) or isinstance(other, DirectionalSpectrum):
#             grid_type = DirectionalSpectrum
#         else:
#             grid_type = Grid

#         return grid_type(
#             self._freq,
#             self._dirs,
#             self._vals * other._vals,
#             freq_hz=False,
#             degrees=False,
#             **self.wave_convention
#         )


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

    def copy(self):
        """Return a copy of the object."""
        return copy.deepcopy(self)

    def rotate(self, angle, degrees=False):
        """
        Rotate the underlying coordinate system a given angle.

        All directions are converted so that:

            dirs_new = dirs_old - angle

        Note that the direction of positive rotation follows the set 'wave_convention'.

        Parameters
        ----------
        angle : float
            Rotation angle.
        degrees : bool
            Weather the rotation angle is given in degrees. If ``False``, radians
            is assumed.

        Returns
        -------
        obj :
            A rotated copy of the object.
        """
        if degrees:
            angle = (np.pi / 180.0) * angle

        new = self.copy()
        dirs_new = (new._dirs - angle) % (2.0 * np.pi)
        new._dirs, new._vals = new._sort(dirs_new, new._vals)
        return new

    def __call__(self, freq_hz=True, degrees=True):
        """
        Return a copy of the grid object's frequencies, directions and values.

        Parameters
        ----------
        freq_hz : bool
            If frequencies should be returned in Hz. If ``False``, rad/s is used.
        degrees : bool
            If directions should be returned in degrees.

        Return
        ------
        freq : 1D-array
            Frequency bins.
        dirs : 1D-array
            Direction bins.
        vals : 2D-array (N, M)
            Value grid as 2-D array of shape (N, M), such that
            ``N=len(freq)`` and ``M=len(dirs)``.
        """
        freq = self._freq.copy()
        dirs = self._dirs.copy()
        vals = self._vals.copy()

        if freq_hz:
            freq = 1.0 / (2.0 * np.pi) * freq

        if degrees:
            dirs = (180.0 / np.pi) * dirs

        return freq, dirs, vals

    def _interpolate_function(self, complex_convert="rectangular", **kw):
        """
        Interpolation function based on ``scipy.interpolate.interp2d``.
        """
        xp = np.concatenate(
            (self._dirs[-1:] - 2 * np.pi, self._dirs, self._dirs[:1] + 2.0 * np.pi)
        )
        yp = self._freq
        zp = np.concatenate(
            (
                self._vals[:, -1:],
                self._vals,
                self._vals[:, :1],
            ),
            axis=1,
        )

        if np.all(np.isreal(zp)):
            return interp2d(xp, yp, zp, **kw)
        elif complex_convert.lower() == "polar":
            amp, phase = complex_to_polar(zp, phase_degrees=False)
            interp_amp = interp2d(xp, yp, amp, **kw)
            interp_phase = interp2d(xp, yp, phase, **kw)
            return lambda *args, **kwargs: (
                polar_to_complex(
                    interp_amp(*args, **kwargs),
                    interp_phase(*args, **kwargs),
                    phase_degrees=False,
                )
            )
        else:
            interp_real = interp2d(xp, yp, np.real(zp), **kw)
            interp_imag = interp2d(xp, yp, np.imag(zp), **kw)
            return lambda *args, **kwargs: (
                interp_real(*args, **kwargs) + 1j * interp_imag(*args, **kwargs)
            )

    def interpolate(
        self,
        freq,
        dirs,
        freq_hz=True,
        degrees=True,
        complex_convert="rectangular",
        fill_value=0.0,
    ):
        """
        Interpolate (linear) the grid values for given frequencies and directions.

        Zero is used as fill value for extrapolation (i.e. `freq` outside the bounds
        of the provided 2D spectrum). Directions are treated as periodic.

        Parameters
        ----------
        freq : array-like
            Frequency bins. Positive and monotonically increasing.
        dirs : array-like
            Direction bins. Positive and monotonically increasing. Must cover the directional
            range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
        freq_hz : bool
            If frequency is given in Hz. If ``False``, rad/s is assumed.
        degrees : bool
            If direction is given in degrees. If ``False``, radians are assumed.
        complex_convert : str, optional
            How to convert complex numbers (if they are present) before interpolating.
            Should be 'rectangular' or 'polar'. If 'rectangular' (default), complex
            values are converted to rectangular form before interpolating. If 'polar',
            the values are instead converted to polar form before interpolating.
            The interpolated values are converted back to complex form before they
            are returned.
        fill_value : float or None
            The value used for extrapolation (i.e., `freq` outside the bounds of
            the provided grid). If ``None``, values outside the domain are extrapolated
            via nearest-neighbor extrapolation. Note that directions are treated
            as periodic.
        """
        freq = np.asarray_chkfinite(freq).copy()
        dirs = np.asarray_chkfinite(dirs).copy()

        if freq_hz:
            freq = 2.0 * np.pi * freq

        if degrees:
            dirs = (np.pi / 180.0) * dirs

        self._check_freq(freq)
        self._check_dirs(dirs)

        interp_fun = self._interpolate_function(
            complex_convert=complex_convert, kind="linear", fill_value=fill_value
        )

        return interp_fun(dirs, freq, assume_sorted=True)

    def reshape(self, freq, dirs, freq_hz=True, degrees=True, complex_convert=None):
        """
        The object's values are (linear) interpolated to match the provided grid.

        Parameters
        ----------
        freq : array-like
            New frequency bins. Positive and monotonically increasing.
        dirs : array-like
            New direction bins. Positive and monotonically increasing. Must cover
            the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
        freq_hz : bool
            If frequency is given in Hz. If ``False``, rad/s is assumed.
        degrees : bool
            If direction is given in degrees. If ``False``, radians are assumed.
        complex_convert : str, optional
            How to convert complex numbers (if they are present) before interpolating.
            Should be 'rectangular' or 'polar'. If 'rectangular' (default), complex
            values are converted to rectangular form before interpolating. If 'polar',
            the values are instead converted to polar form before interpolating.
            The interpolated values are converted back to complex form before they
            are returned.

        Returns
        -------
        obj :
            Reshaped object.
        """
        freq_new = np.asarray_chkfinite(freq).copy()
        dirs_new = np.asarray_chkfinite(dirs).copy()

        if freq_hz:
            freq_new = 2.0 * np.pi * freq_new

        if degrees:
            dirs_new = (np.pi / 180.0) * dirs_new

        self._check_freq(freq_new)
        self._check_dirs(dirs_new)

        vals_new = self.interpolate(
            freq_new,
            dirs_new,
            freq_hz=False,
            degrees=False,
            complex_convert=complex_convert,
        )
        new = self.copy()
        new._freq, new._dirs, new._vals = freq_new, dirs_new, vals_new
        return new

    def __mul__(self, other):
        """
        Multiply with another Grid object.

        Both grids must have the same frequency/direction bins.

        Parameters
        ----------
        other : obj
            Grid object.

        Returns
        -------
        obj :
            New Grid object.
        """
        if not isinstance(other, Grid):
            raise ValueError()
        elif np.any(self._freq != other._freq) or np.any(self._dirs != other._dirs):
            raise ValueError()
        elif self.wave_convention != other.wave_convention:
            raise ValueError()

        return Grid(
            self._freq,
            self._dirs,
            self._vals * other._vals,
            freq_hz=False,
            degrees=False,
            **self.wave_convention
        )


class RAO(Grid):
    """
    RAO.
    """

    @classmethod
    def from_amp_phase(
        cls,
        freq,
        dirs,
        amp,
        phase,
        freq_hz=True,
        degrees=True,
        phase_degrees=False,
        clockwise=True,
        waves_coming_from=True,
    ):
        """
        Alternative constructor.
        """

        rao_complex = polar_to_complex(amp, phase, phase_degrees=phase_degrees)

        return cls(
            freq,
            dirs,
            rao_complex,
            freq_hz=freq_hz,
            degrees=degrees,
            clockwise=clockwise,
            waves_coming_from=waves_coming_from,
        )

    def __mul__(self, other):
        """
        Multiply with another Grid object.

        Both grids must have the same frequency/direction bins.

        Parameters
        ----------
        other : obj
            Grid object.

        Returns
        -------
        obj :
            New Grid object.
        """
        new = super().__mul__(other)
        if isinstance(other, DirectionalSpectrum):
            return DirectionalSpectrum(new._freq, new._dirs, new._vals, **new.wave_convention)
        else:
            return new

    def conjugate(self):
        """
        Complex conjugate.
        """
        new = self.copy()
        new._vals = new._vals.conjugate()
        return new


class DirectionalSpectrum(Grid):
    def __mul__(self, other):
        """
        Multiply with another Grid object.

        Both grids must have the same frequency/direction bins.

        Parameters
        ----------
        other : obj
            Grid object.

        Returns
        -------
        obj :
            New DirectionalSpectrum object.
        """
        new = super().__mul__(other)
        return DirectionalSpectrum(new._freq, new._dirs, new._vals, **new.wave_convention)
