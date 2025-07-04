import copy
import warnings
from abc import ABC, abstractmethod
from numbers import Number

import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root_scalar
from scipy.special import gamma

from ._utils import _robust_modulus, complex_to_polar, polar_to_complex


def _check_is_similar(*grids, exact_type=True):
    """
    Check if grid objects are similar.
    """
    grids = list(grids)
    grid_ref = grids.pop()

    if exact_type:
        type_ = type(grid_ref)

        def check_type(grid, grid_type):
            return type(grid) is grid_type

    else:
        type_ = Grid
        check_type = isinstance

    for grid_i in grids:
        if not check_type(grid_i, type_):
            raise TypeError("Object types are not similar.")
        elif grid_ref._vals.shape != grid_i._vals.shape:
            raise ValueError("Grid objects have different shape.")
        elif not np.allclose(
            grid_ref._freq, grid_i._freq, atol=1e-8, rtol=0.0
        ) or not np.allclose(grid_ref._dirs, grid_i._dirs, atol=1e-8, rtol=0.0):
            raise ValueError(
                "Grid objects have different frequency/direction coordinates."
            )
        elif grid_ref.wave_convention != grid_i.wave_convention:
            raise ValueError("Grid objects have different wave conventions.")


def multiply(grid1, grid2, output_type="Grid"):
    """
    Multiply values (element-wise).

    Parameters
    ----------
    grid1 : obj
        Grid object.
    grid2 : obj
        Grid object.
    output_type : {'Grid', 'RAO', 'DirectionalSpectrum', 'WaveSpectrum', 'DirectionalBinSpectrum', 'WaveBinSpectrum'}
        Output grid type.
    """

    TYPE_MAP = {
        "Grid": Grid,
        "RAO": RAO,
        "DirectionalSpectrum": DirectionalSpectrum,
        "DirectionalBinSpectrum": DirectionalBinSpectrum,
        "WaveSpectrum": WaveSpectrum,
        "WaveBinSpectrum": WaveBinSpectrum,
        "grid": Grid,  # for backward compatibility
        "rao": RAO,  # for backward compatibility
        "directional_spectrum": DirectionalSpectrum,  # for backward compatibility
        "wave_spectrum": WaveSpectrum,  # for backward compatibility
    }

    output_type_ = TYPE_MAP.get(output_type, output_type)

    if not (isinstance(output_type_, type) and issubclass(output_type_, Grid)):
        raise ValueError(f"Invalid `output_type`: {output_type_!r}")

    _check_is_similar(grid1, grid2, exact_type=False)

    freq = grid1._freq
    dirs = grid1._dirs
    vals = np.multiply(grid1._vals, grid2._vals)
    convention = grid1.wave_convention

    new = Grid(
        freq,
        dirs,
        vals,
        freq_hz=False,
        degrees=False,
        **convention,
    )

    return output_type_.from_grid(new)


def _cast_to_grid(grid):
    """
    Cast Grid-like object to ``Grid`` type.

    Note that this type conversion may lead to loss of information/functionality
    for derived classes.
    """
    new = Grid(
        *grid.grid(freq_hz=grid._freq_hz, degrees=grid._degrees),
        freq_hz=grid._freq_hz,
        degrees=grid._degrees,
        **grid.wave_convention,
    )

    return new


def _check_foldable(dirs, degrees=False, sym_plane="xz"):
    """Checks that directions can be folded about a given symmetry plane"""
    dirs = np.asarray_chkfinite(dirs).copy()

    if len(dirs) == 0:
        raise ValueError("`rao` is defined only at the bounds. Nothing to mirror.")

    if degrees:
        dirs = dirs * np.pi / 180.0

    if sym_plane.lower() == "xz":
        dirs_bools = np.sin(dirs) >= 0.0
        error_msg = (
            "`rao` must be defined in the range [0, 180] degrees or [180, 360) degrees."
        )
    elif sym_plane.lower() == "yz":
        dirs_bools = np.cos(dirs) >= 0.0
        error_msg = (
            "`rao` must be defined in the range [90, 270] degrees or [270, 90] degrees."
        )
    else:
        raise ValueError()

    if not (all(dirs_bools) or all(~dirs_bools)):
        raise ValueError(error_msg)


def _sort(dirs, vals):
    """
    Sort directions and values according to (unsorted) directions.
    """
    dirs = np.asarray_chkfinite(dirs)
    vals = np.asarray_chkfinite(vals)
    sorted_args = np.argsort(dirs)
    return dirs[sorted_args], vals[:, sorted_args]


class _GridInterpolator:
    """
    Interpolation function based on ``scipy.interpolate.RegularGridInterpolator``.
    """

    def __init__(self, freq, dirs, vals, complex_convert="rectangular", **kwargs):
        xp = np.concatenate((dirs[-1:] - 2 * np.pi, dirs, dirs[:1] + 2.0 * np.pi))

        yp = freq
        zp = np.concatenate(
            (
                vals[:, -1:],
                vals,
                vals[:, :1],
            ),
            axis=1,
        )

        if np.all(np.isreal(zp)):
            self._interpolate = RGI((xp, yp), zp.T, **kwargs)
        elif complex_convert.lower() == "polar":
            amp, phase = complex_to_polar(zp, phase_degrees=False)
            phase_complex = np.cos(phase) + 1j * np.sin(phase)
            interp_amp = RGI((xp, yp), amp.T, **kwargs)
            interp_phase = RGI((xp, yp), phase_complex.T, **kwargs)
            self._interpolate = lambda *args_, **kwargs_: (
                polar_to_complex(
                    interp_amp(*args_, **kwargs_),
                    np.angle(interp_phase(*args_, **kwargs_)),
                    phase_degrees=False,
                )
            )
        elif complex_convert.lower() == "rectangular":
            interp_real = RGI((xp, yp), np.real(zp.T), **kwargs)
            interp_imag = RGI((xp, yp), np.imag(zp.T), **kwargs)
            self._interpolate = lambda *args_, **kwargs_: (
                interp_real(*args_, **kwargs_) + 1j * interp_imag(*args_, **kwargs_)
            )
        else:
            raise ValueError("Unknown 'complex_convert' type")

    def __call__(self, freq, dirs):
        dirsnew, freqnew = np.meshgrid(dirs, freq, indexing="ij", sparse=True)
        return self._interpolate((dirsnew, freqnew)).T


def mirror(rao, dof, sym_plane="xz"):
    """
    Mirrors/folds an RAO object about a symmetry plane.

    Requires that the RAO is defined for directions that allow folding with the
    given symmetry plane. I.e., folding about the xz-plane requires that the RAO
    is defined for directions in the range [0, 180] degrees or [180, 360] degrees.
    Similarly, folding about the yz-plane requires that the RAO is defined for directions
    in the range [90, 270] degrees or [270, 90] degrees.

    Parameters
    ----------
    rao : RAO
        RAO object.
    dof : {'surge', 'sway', 'heave', 'roll', 'pitch', 'yaw'}
        Which degree-of-freedom the RAO object represents.
    sym_plane : {'xz', 'yz'}
        Symmetry plane, determining which axis to mirror the RAO about.

    Returns
    -------
    rao : RAO
        Extended (mirrored) RAO object.

    Examples
    --------
    If you have an RAO defined only in half the directional domain (e.g., [0, 180] degrees),
    you can mirror it once about a symmetry plane to obtain the 'full' RAO, defined over
    the whole directional domain:

    >>> # Symmetry about xz-plane
    >>> rao_full = wr.mirror(rao, "heave", sym_plane="xz")

    If you have an RAO defined only in one quadrant (e.g., [0, 90] degrees), you
    can mirror it twise to obtain the 'full' RAO, defined over the whole directional
    domain:

    >>> # Symmetry about xz- and yz-plane
    >>> rao_full = wr.mirror(
    ...     wr.mirror(rao, "heave", sym_plane="xz"),
    ...     "heave",
    ...     sym_plane="yz"
    ... )
    """

    sym_plane = sym_plane.lower()
    dof = dof.lower()
    freq, dirs, vals = rao.grid()

    if dof not in ("surge", "sway", "heave", "roll", "pitch", "yaw"):
        raise ValueError(
            "`dof` must be 'surge', 'sway', 'heave', 'roll', 'pitch' or 'yaw'"
        )

    if rao._degrees:
        periodicity = 360.0
    else:
        periodicity = 2 * np.pi

    scale_phase = 1
    if sym_plane == "xz":
        bounds = (0.0, periodicity / 2.0)
        if dof in ("sway", "roll", "yaw"):
            scale_phase = -1
    elif sym_plane == "yz":
        bounds = (periodicity / 4.0, 3.0 * periodicity / 4.0)
        if dof in ("surge", "pitch", "yaw"):
            scale_phase = -1
    else:
        raise ValueError("`sym_plane` should be 'xz' or 'yz'")

    lb_0, ub_0 = np.nextafter(bounds[0], (-periodicity, periodicity))
    lb_1, ub_1 = np.nextafter(bounds[1], (-periodicity, periodicity))
    exclude_bounds = ((dirs >= ub_0) | (dirs <= lb_0)) & (
        (dirs >= ub_1) | (dirs <= lb_1)
    )

    _check_foldable(dirs[exclude_bounds], degrees=rao._degrees, sym_plane=sym_plane)

    vals_folded = scale_phase * vals[:, exclude_bounds]
    if sym_plane == "xz":
        dirs_folded = -1 * dirs[exclude_bounds]
    elif sym_plane == "yz":
        dirs_folded = -1 * dirs[exclude_bounds] + periodicity / 2.0

    vals_mirrored = np.concatenate((vals, vals_folded), axis=1)
    dirs_mirrored = np.concatenate((dirs, dirs_folded))
    dirs_mirrored = _robust_modulus(dirs_mirrored, periodicity)
    dirs_mirrored, vals_mirrored = _sort(dirs_mirrored, vals_mirrored)

    return RAO(
        freq,
        dirs_mirrored,
        vals_mirrored,
        degrees=rao._degrees,
        freq_hz=rao._freq_hz,
        **rao.wave_convention,
    )


class Grid:
    """
    Two-dimentional frequency/(wave)direction grid.

    Parameters
    ----------
    freq : array-like
        1-D array of grid frequency coordinates. Positive and monotonically increasing.
    dirs : array-like
        1-D array of grid direction coordinates. Positive and monotonically increasing.
        Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
    vals : array-like (N, M)
        Values associated with the grid. Should be a 2-D array of shape (N, M),
        such that ``N=len(freq)`` and ``M=len(dirs)``.
    freq_hz : bool
        If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
    degrees : bool
        If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
    clockwise : bool
        If positive directions are defined to be 'clockwise' (``True``) or 'counterclockwise'
        (``False``). Clockwise means that the directions follow the right-hand rule
        with an axis pointing downwards.
    waves_coming_from : bool
        If waves are 'coming from' the given directions. If ``False``, 'going towards'
        convention is assumed.
    """

    def __init__(
        self,
        freq,
        dirs,
        vals,
        freq_hz=False,
        degrees=False,
        clockwise=False,
        waves_coming_from=True,
    ):
        self._freq = np.asarray_chkfinite(freq).copy()
        self._dirs = np.asarray_chkfinite(dirs).copy()
        self._vals = np.asarray_chkfinite(vals).copy()
        self._clockwise = clockwise
        self._waves_coming_from = waves_coming_from
        self._freq_hz = freq_hz
        self._degrees = degrees

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

    def __repr__(self):
        return "Grid"

    @classmethod
    def from_grid(cls, grid):
        """
        Construct from a Grid-like object.

        Parameters
        ----------
        grid : obj
            Grid object.

        Returns
        -------
        cls :
            Initialized object.
        """
        return cls(
            *grid.grid(freq_hz=grid._freq_hz, degrees=grid._degrees),
            freq_hz=grid._freq_hz,
            degrees=grid._degrees,
            **grid.wave_convention,
        )

    def _check_freq(self, freq):
        """
        Check frequency bins.
        """
        if freq.ndim != 1:
            raise ValueError("`freq` must be 1 dimensional.")

        if np.any(freq[:-1] >= freq[1:]) or freq[0] < 0:
            raise ValueError(
                "Frequencies must be positive and monotonically increasing."
            )

    def _check_dirs(self, dirs):
        """
        Check direction bins.
        """
        if dirs.ndim != 1:
            raise ValueError("`dirs` must be 1 dimensional.")

        if np.any(dirs[:-1] >= dirs[1:]) or dirs[0] < 0 or dirs[-1] >= 2.0 * np.pi:
            raise ValueError(
                "Directions must be positive, monotonically increasing, and "
                "be [0., 360.) degs (or [0., 2*pi) rads)."
            )

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

    def dirs(self, degrees=None):
        """
        Direction coordinates.

        Parameters
        ----------
        degrees : bool
            If directions should be returned in 'degrees'. If ``False``, 'radians'
            is used. Defaults to original units used during initialization.
        """
        dirs = self._dirs.copy()

        if degrees is None:
            degrees = self._degrees

        if degrees:
            dirs = (180.0 / np.pi) * dirs

        return dirs

    def grid(self, freq_hz=None, degrees=None):
        """
        Return a copy of the object's frequency/direction coordinates and corresponding
        grid values.

        Parameters
        ----------
        freq_hz : bool
            If frequencies should be returned in 'Hz'. If ``False``, 'rad/s' is used.
            Defaults to original units used during initialization.
        degrees : bool
            If directions should be returned in 'degrees'. If ``False``, 'radians'
            is used. Defaults to original units used during initialization.

        Returns
        -------
        freq : array
            1-D array of grid frequency coordinates.
        dirs : array
            1-D array of grid direction coordinates.
        vals : array (N, M)
            Grid values as 2-D array of shape (N, M), such that ``N=len(freq)``
            and ``M=len(dirs)``.
        """
        freq = self.freq(freq_hz=freq_hz)
        dirs = self.dirs(degrees=degrees)
        vals = self._vals.copy()

        return freq, dirs, vals

    @property
    def wave_convention(self):
        """
        Wave direction convention.
        """
        return {
            "clockwise": self._clockwise,
            "waves_coming_from": self._waves_coming_from,
        }

    def set_wave_convention(self, clockwise=True, waves_coming_from=True):
        """
        Set wave direction convention.

        Directions and values will be converted (in-place) to the given convention.

        Parameters
        ----------
        clockwise : bool
            If positive directions are defined to be 'clockwise' (``True``) or
            'counterclockwise' (``False``). Clockwise means that the directions
            follow the right-hand rule with an axis pointing downwards.
        waves_coming_from : bool
            If waves are 'coming from' the given directions. If ``False``, 'going towards'
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
        dirs_new, vals_new = _sort(dirs_new, vals_org)

        return freq_new, dirs_new, vals_new

    @staticmethod
    def _convert_dirs(dirs, config_new, config_org, degrees=False):
        """
        Convert wave directions from one convention to another.

        Parameters
        ----------
        dirs : float or array-like
            Wave directions in 'radians' expressed according to 'original' convention.
        config_new : dict
            New wave direction convention.
        config_org : dict
            Original wave direction convention.
        degrees : bool
            If directions are given in 'degrees'. If ``False``, 'radians' is assumed.

        Return
        ------
        dirs : numpy.array
            Wave directions in 'radians' expressed according to 'new' convention.
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

        return _robust_modulus(dirs, periodicity)

    def copy(self):
        """Return a copy of the object."""
        return copy.deepcopy(self)

    def rotate(self, angle, degrees=False):
        """
        Rotate the underlying grid coordinate system a given angle.

        All directions are converted so that:

            dirs_new = dirs_old - angle

        Note that the direction of positive rotation follows the set 'wave convention'.

        Parameters
        ----------
        angle : float
            Rotation angle.
        degrees : bool
            Whether the rotation angle is given in 'degrees'. If ``False``, 'radians'
            is assumed.

        Returns
        -------
        obj :
            A copy of the object where the underlying coordinate system is rotated.
        """
        if degrees:
            angle = (np.pi / 180.0) * angle

        new = self.copy()
        dirs_new = _robust_modulus(new._dirs - angle, 2.0 * np.pi)
        new._dirs, new._vals = _sort(dirs_new, new._vals)
        return new

    def __mul__(self, other):
        """
        Multiply values (element-wise).

        Both grids must have the same frequency/direction coordinates, and the same
        'wave convention'.

        Parameters
        ----------
        other : obj or numeric
            Grid object or number to be multiplied with.

        Returns
        -------
        obj :
            A copy of the object where the values are multiplied with values of
            another grid.
        """
        new = self.copy()

        if isinstance(other, Number):
            new._vals = new._vals * other
        else:
            _check_is_similar(self, other, exact_type=True)
            new._vals = new._vals * other._vals

        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        """
        Add values (element-wise).

        Both grids must have the same frequency/direction coordinates, and the same
        'wave convention'.

        Parameters
        ----------
        other : obj or numeric
            Grid object or number to be added.

        Returns
        -------
        obj :
            A copy of the object where the values are added with another grid's values.
        """
        new = self.copy()

        if isinstance(other, Number):
            new._vals = new._vals + other
        else:
            _check_is_similar(self, other, exact_type=True)
            new._vals = new._vals + other._vals

        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract values (element-wise).

        Both grids must have the same frequency/direction coordinates, and the same
        'wave convention'.

        Parameters
        ----------
        other : obj or numeric
            Grid object or number to be subtracted.

        Returns
        -------
        obj :
            A copy of the object where the values are subtracted with another grid's values.
        """
        new = self.copy()

        if isinstance(other, Number):
            new._vals = new._vals - other
        else:
            _check_is_similar(self, other, exact_type=True)
            new._vals = new._vals - other._vals

        return new

    def __rsub__(self, other):
        return self.__sub__(other)

    def conjugate(self):
        """
        Return a copy of the object with complex conjugate values.
        """
        new = self.copy()
        new._vals = new._vals.conjugate()
        return new

    @property
    def real(self):
        """
        Return a new Grid object where all values are converted to their real part.
        """
        new = _cast_to_grid(self)
        new._vals = new._vals.real
        return new

    @property
    def imag(self):
        """
        Return a new Grid object where all values are converted to their imaginary part.
        """
        new = _cast_to_grid(self)
        new._vals = new._vals.imag
        return new

    def interpolate(
        self,
        freq,
        dirs,
        freq_hz=False,
        degrees=False,
        complex_convert="rectangular",
        fill_value=0.0,
    ):
        """
        Interpolate (linear) the grid values to match the given frequency and direction
        coordinates.

        A 'fill value' is used for extrapolation (i.e. `freq` outside the bounds
        of the provided 2-D grid). Directions are treated as periodic.

        Parameters
        ----------
        freq : array-like
            1-D array of grid frequency coordinates. Positive and monotonically increasing.
        dirs : array-like
            1-D array of grid direction coordinates. Positive and monotonically increasing.
        freq_hz : bool
            If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
        degrees : bool
            If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
        complex_convert : str, optional
            How to convert complex number grid values before interpolating. Should
            be 'rectangular' or 'polar'. If 'rectangular' (default), complex values
            are converted to rectangular form (i.e., real and imaginary part) before
            interpolating. If 'polar', the values are instead converted to polar
            form (i.e., amplitude and phase) before interpolating. The values are
            converted back to complex form after interpolation.
        fill_value : float or None
            The value used for extrapolation (i.e., `freq` outside the bounds of
            the provided grid). If ``None``, values outside the frequency domain
            are extrapolated via nearest-neighbor extrapolation. Note that directions
            are treated as periodic (and will not need extrapolation).

        Returns
        -------
        array :
            Interpolated grid values.
        """
        warnings.warn(
            "The `Grid.interpolate` method is deprecated and will be removed in a future release. ",
            DeprecationWarning,
            stacklevel=2,
        )

        freq = np.asarray_chkfinite(freq).reshape(-1)
        dirs = np.asarray_chkfinite(dirs).reshape(-1)

        if freq_hz:
            freq = 2.0 * np.pi * freq

        if degrees:
            dirs = (np.pi / 180.0) * dirs

        self._check_freq(freq)
        self._check_dirs(dirs)

        interp_fun = _GridInterpolator(
            self._freq,
            self._dirs,
            self._vals,
            complex_convert=complex_convert,
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

        return interp_fun(freq, dirs)

    def reshape(
        self,
        freq,
        dirs,
        freq_hz=False,
        degrees=False,
        complex_convert="rectangular",
        fill_value=0.0,
    ):
        """
        Reshape the grid to match the given frequency/direction coordinates. Grid
        values will be interpolated (linear).

        Parameters
        ----------
        freq : array-like
            1-D array of new grid frequency coordinates. Positive and monotonically
            increasing.
        dirs : array-like
            1-D array of new grid direction coordinates. Positive and monotonically increasing.
            Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
        freq_hz : bool
            If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
        degrees : bool
            If direction is given in 'degrees'. If ``False``, 'radians' are assumed.
        complex_convert : str, optional
            How to convert complex number grid values before interpolating. Should
            be 'rectangular' or 'polar'. If 'rectangular' (default), complex values
            are converted to rectangular form (i.e., real and imaginary part) before
            interpolating. If 'polar', the values are instead converted to polar
            form (i.e., amplitude and phase) before interpolating. The values are
            converted back to complex form after interpolation.
        fill_value : float or None
            The value used for extrapolation (i.e., `freq` outside the bounds of
            the provided grid). If ``None``, values outside the frequency domain
            are extrapolated via nearest-neighbor extrapolation. Note that directions
            are treated as periodic (and will not need extrapolation).

        Returns
        -------
        obj :
            A copy of the object where the underlying coordinate system is reshaped.
        """
        warnings.warn(
            "The `Grid.reshape` method is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

        freq_new = np.asarray_chkfinite(freq).copy()
        dirs_new = np.asarray_chkfinite(dirs).copy()

        if freq_hz:
            freq_new = 2.0 * np.pi * freq_new

        if degrees:
            dirs_new = (np.pi / 180.0) * dirs_new

        self._check_freq(freq_new)
        self._check_dirs(dirs_new)

        interp_fun = _GridInterpolator(
            self._freq,
            self._dirs,
            self._vals,
            complex_convert=complex_convert,
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

        vals_new = interp_fun(freq_new, dirs_new)
        new = self.copy()
        new._freq, new._dirs, new._vals = freq_new, dirs_new, vals_new
        return new


class DisableComplexMixin:
    @property
    def imag(self):
        raise AttributeError(f"'{self}' object has no attribute 'imag'.")

    @property
    def real(self):
        raise AttributeError(f"'{self}' object has no attribute 'real'.")

    @property
    def conjugate(self):
        raise AttributeError(f"'{self}' object has no attribute 'conjugate'.")


class RAO(Grid):
    """
    Response amplitude operator (RAO).

    The ``RAO`` class extends the :class:`~waveresponse.Grid` class, and is a
    two-dimensional frequency/(wave)direction grid. The RAO values represents a
    transfer function that can be used to calculate a degree-of-freedom's response
    based on a 2-D wave spectrum.

    Parameters
    ----------
    freq : array-like
        1-D array of grid frequency coordinates. Positive and monotonically increasing.
    dirs : array-like
        1-D array of grid direction coordinates. Positive and monotonically increasing.
        Should be within the directional range [0, 360) degrees (or [0, 2*pi) radians).
        See Notes.
    vals : array-like (N, M)
        RAO values (complex) associated with the grid. Should be a 2-D array of shape (N, M),
        such that ``N=len(freq)`` and ``M=len(dirs)``.
    freq_hz : bool
        If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
    degrees : bool
        If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
    clockwise : bool
        If positive directions are defined to be 'clockwise' (``True``) or 'counterclockwise'
        (``False``). Clockwise means that the directions follow the right-hand rule
        with an axis pointing downwards.
    waves_coming_from : bool
        If waves are 'coming from' the given directions. If ``False``, 'going towards'
        convention is assumed.

    Notes
    -----
    If the RAO is going to be used in response calculation (where the RAO is combined
    with a wave spectrum to form a response spectrum), it is important that the
    RAO covers the full directional domain, i.e., [0, 360) degrees, with a sufficient
    resolution. If the RAO object only partly covers the directional domain, you
    can consider to use the :func:`mirror` function to 'expand' the RAO with values
    that are folded about a symmetry plane.

    """

    def __init__(
        self,
        freq,
        dirs,
        vals,
        freq_hz=False,
        degrees=False,
        clockwise=False,
        waves_coming_from=True,
    ):
        super().__init__(
            freq,
            dirs,
            vals,
            freq_hz=freq_hz,
            degrees=degrees,
            clockwise=clockwise,
            waves_coming_from=waves_coming_from,
        )
        self._phase_degrees = False
        self._phase_leading = True

    @classmethod
    def from_amp_phase(
        cls,
        freq,
        dirs,
        amp,
        phase,
        phase_degrees=False,
        phase_leading=True,
        freq_hz=True,
        degrees=True,
        clockwise=True,
        waves_coming_from=True,
    ):
        """
        Construct an ``RAO`` object from amplitude and phase values.

        Note that the RAO is converted to, and stored as, complex values internally.

        Parameters
        ----------
        freq : array-like
            1-D array of grid frequency coordinates. Positive and monotonically increasing.
        dirs : array-like
            1-D array of grid direction coordinates. Positive and monotonically increasing.
            Should be within the directional range [0, 360) degrees (or [0, 2*pi) radians).
            See Notes.
        amp : array-like (N, M)
            RAO amplitude values associated with the grid. Should be a 2-D array
            of shape (N, M), such that ``N=len(freq)`` and ``M=len(dirs)``.
        phase : array-like (N, M)
            RAO phase values associated with the grid. Should be a 2-D array
            of shape (N, M), such that ``N=len(freq)`` and ``M=len(dirs)``.
        phase_degrees : bool
            If the RAO phase values are given in 'degrees'. If ``False``, 'radians'
            is assumed.
        phase_leading : bool
            Whether the phase values follow the 'leading' convention (``True``)
            or the 'lagging' convention (``False``). Mathematically, an RAO with
            phase lead convention is expressed as a complex number of the form
            ``A * exp(j * phi)``, where ``A`` represents the amplitude and ``phi``
            represents the phase angle. Whereas an RAO with phase lag convention
            is expressed as ``A * exp(-j * phi)``.
        freq_hz : bool
            If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
        degrees : bool
            If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
        clockwise : bool
            If positive directions are defined to be 'clockwise'. If ``False``,
            'counterclockwise' is assumed.
        waves_coming_from : bool
            If waves are 'coming from' the given directions. If ``False``, 'going towards'
            convention is assumed.

        Returns
        -------
        obj :
            Initialized RAO object.

        Notes
        -----
        If the RAO is going to be used in response calculation (where the RAO is combined
        with a wave spectrum to form a response spectrum), it is important that the
        RAO covers the full directional domain, i.e., [0, 360) degrees, with a sufficient
        resolution. If the RAO object only partly covers the directional domain, you
        can consider to use the :func:`mirror` function to 'expand' the RAO with values
        that are folded about a symmetry plane.

        """
        if not phase_leading:
            phase = -np.asarray_chkfinite(phase)

        rao_complex = polar_to_complex(amp, phase, phase_degrees=phase_degrees)

        rao = cls(
            freq,
            dirs,
            rao_complex,
            freq_hz=freq_hz,
            degrees=degrees,
            clockwise=clockwise,
            waves_coming_from=waves_coming_from,
        )
        rao._phase_degrees = phase_degrees
        rao._phase_leading = phase_leading
        return rao

    def reshape(
        self,
        freq,
        dirs,
        freq_hz=False,
        degrees=False,
        complex_convert="rectangular",
        fill_value=0.0,
    ):
        """
        Reshape the grid to match the given frequency/direction coordinates. Grid
        values will be interpolated (linear).

        Parameters
        ----------
        freq : array-like
            1-D array of new grid frequency coordinates. Positive and monotonically
            increasing.
        dirs : array-like
            1-D array of new grid direction coordinates. Positive and monotonically increasing.
            Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
        freq_hz : bool
            If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
        degrees : bool
            If direction is given in 'degrees'. If ``False``, 'radians' are assumed.
        complex_convert : str, optional
            How to convert complex number grid values before interpolating. Should
            be 'rectangular' or 'polar'. If 'rectangular' (default), complex values
            are converted to rectangular form (i.e., real and imaginary part) before
            interpolating. If 'polar', the values are instead converted to polar
            form (i.e., amplitude and phase) before interpolating. The values are
            converted back to complex form after interpolation.
        fill_value : float or None
            The value used for extrapolation (i.e., `freq` outside the bounds of
            the provided grid). If ``None``, values outside the frequency domain
            are extrapolated via nearest-neighbor extrapolation. Note that directions
            are treated as periodic (and will not need extrapolation).

        Returns
        -------
        obj :
            A copy of the object where the underlying coordinate system is reshaped.
        """

        freq_new = np.asarray_chkfinite(freq).copy()
        dirs_new = np.asarray_chkfinite(dirs).copy()

        if freq_hz:
            freq_new = 2.0 * np.pi * freq_new

        if degrees:
            dirs_new = (np.pi / 180.0) * dirs_new

        self._check_freq(freq_new)
        self._check_dirs(dirs_new)

        interp_fun = _GridInterpolator(
            self._freq,
            self._dirs,
            self._vals,
            complex_convert=complex_convert,
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

        vals_new = interp_fun(freq_new, dirs_new)
        new = self.copy()
        new._freq, new._dirs, new._vals = freq_new, dirs_new, vals_new
        return new

    def differentiate(self, n=1):
        """
        Return the nth derivative of the RAO.

        Parameters
        ----------
        n : int
            Order of differentiation.

        Returns
        -------
        obj :
            Differentiated RAO object.
        """
        new = self.copy()
        new._vals = new._vals * ((1j * new._freq.reshape(-1, 1)) ** n)
        return new

    def to_amp_phase(
        self, phase_degrees=None, phase_leading=None, freq_hz=None, degrees=None
    ):
        """
        Return the RAO as amplitude and phase values.

        Parameters
        ----------
        phase_degrees : bool
            If phase values should be returned in 'degrees'. If ``False``, 'radians'
            is used. Defaults to original units used during initialization or ``False``.
        phase_leading : bool
            Whether the phase values should follow the 'leading' convention (``True``)
            or the 'lagging' convention (``False``). If ``None``, it defaults to
            the convention given during initialization, or the lagging convention
            if no convention was specified during initialization. Mathematically,
            an RAO with phase lead convention is expressed as a complex number of
            the form ``A * exp(j * phi)``, where ``A`` represents the amplitude and
            ``phi`` represents the phase angle. Whereas an RAO with phase lag convention
            is expressed as ``A * exp(-j * phi)``.
        freq_hz : bool
            If frequencies should be returned in 'Hz'. If ``False``, 'rad/s' is used.
            Defaults to original units used during initialization.
        degrees : bool
            If directions should be returned in 'degrees'. If ``False``, 'radians'
            is used. Defaults to original units used during initialization.

        Returns
        -------
        freq : array
            1-D array of grid frequency coordinates.
        dirs : array
            1-D array of grid direction coordinates.
        amp : array (N, M)
            RAO amplitude values as 2-D array of shape (N, M), such that ``N=len(freq)``
            and ``M=len(dirs)``.
        phase : array (N, M)
            RAO phase values as 2-D array of shape (N, M), such that ``N=len(freq)``
            and ``M=len(dirs)``.
        """
        if freq_hz is None:
            freq_hz = self._freq_hz
        if degrees is None:
            degrees = self._degrees
        if phase_degrees is None:
            phase_degrees = self._phase_degrees
        if phase_leading is None:
            phase_leading = self._phase_leading

        freq, dirs, vals = self.grid(freq_hz=freq_hz, degrees=degrees)
        vals_amp, vals_phase = complex_to_polar(vals, phase_degrees=False)

        if not phase_leading:
            vals_phase = -vals_phase
            vals_phase = np.where(np.isclose(vals_phase, -np.pi), np.pi, vals_phase)

        if phase_degrees:
            vals_phase = (180.0 / np.pi) * vals_phase

        return freq, dirs, vals_amp, vals_phase

    def __repr__(self):
        return "RAO"


class _SpectrumMixin:
    """
    Mixin class Spectrum methods.
    """

    @abstractmethod
    def _freq_spectrum(self, freq_hz=None):
        """
        Integrate the spectrum over the directional domain to obtain the non-directional
        frequency spectrum.

        Parameters
        ----------
        freq_hz : bool
            If frequencies in 'Hz' should be used. If ``False``, 'rad/s' is used.
            Defaults to original unit used during initialization.

        Returns
        -------
        f : array
            1-D array of frequency coordinates in 'Hz' or 'rad/s'.
        s : array
            1-D array of spectral density values in 'm^2/Hz' or 'm^2/(rad/s)'.
        """
        raise NotImplementedError

    def _dir_spectrum(self, degrees=None):
        """
        Integrate the spectrum over the frequency domain to obtain the 'directional
        distribution' of the spectrum.

        Parameters
        ----------
        degrees : bool
            If directions should be returned in 'degrees'. If ``False``, 'radians'
            is used. Defaults to original unit used during initialization.

        Returns
        -------
        d : array
            1-D array of direction coordinates in 'deg' or 'rad'.
        s : array
            1-D array of spectral density values in 'm^2/deg' or 'm^2/rad'.
        """

        if degrees is None:
            degrees = self._degrees

        f, d, vv = self.grid(freq_hz=False, degrees=degrees)

        s = np.array([trapezoid(vv_f, f) for vv_f in vv.T])

        return d, s

    def spectrum1d(self, axis=1, freq_hz=None, degrees=None):
        """
        Integrate the spectrum over a given axis.

        Parameters
        ----------
        axis : int
            Axis along which integration of the spectrum is done. For `axis=1`
            (default) the spectrum is integrated over direction, resulting
            in the so-called 'non-directional' spectrum. For `axis=0` the
            spectrum is integrated over frequency, resulting in the directional
            'distribution' of the spectrum.
        freq_hz : bool
            If frequencies should be returned in 'Hz'. If ``False``, 'rad/s' is
            used. This option is only relevant if `axis=1`. Defaults to original
            unit used during instantiation.
        degrees : bool
            If directions should be returned in degrees. This option is only
            relevant if `axis=0`. Defaults to original unit used during
            instantiation.

        Returns
        -------
        x : 1-D array
            Spectrum bins corresponding to the specified axis. `axis=1` yields
            frequencies, while `axis=0` yields directions.
        s : 1-D array
            Spectrum density values, where the spectrum is integrated over the
            specified axis.
        """

        if axis == 0:  # integrate over frequency
            return self._dir_spectrum(degrees=degrees)
        elif axis == 1:  # integrate over frequency
            return self._freq_spectrum(freq_hz=freq_hz)
        else:
            raise ValueError("'axis' must be 0 or 1.")

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
        f, s = self._freq_spectrum(freq_hz=freq_hz)
        m_n = trapezoid((f**n) * s, f)
        return m_n

    def var(self):
        """
        Variance (integral) of the spectrum.
        """
        return self.moment(0)

    def std(self):
        """
        Standard deviation of the spectrum.
        """
        return np.sqrt(self.var())

    @property
    def tz(self):
        """
        Mean zero-crossing period, Tz, in 'seconds'. Only applicable for
        positive real-valued spectra.

        Calculated from the zeroth- and second-order spectral moments according to:

        ``tz = sqrt(m0 / m2)``

        where the spectral moments are calculated by integrating over frequency in Hz.

        Notes
        -----
        The mean zero-crossing period is calculated according to Equation (8.33)
        in reference [1].

        References
        ----------
        [1] A. Naess and T. Moan, (2013), "Stochastic dynamics of marine structures",
        Cambridge University Press.

        """
        if np.iscomplex(self._vals).any() or (self._vals < 0.0).any():
            raise ValueError(
                "Mean zero-crossing period is only defined for positive real-valued spectra."
            )

        m0 = self.moment(0, freq_hz=True)
        m2 = self.moment(2, freq_hz=True)

        return np.sqrt(m0 / m2)

    def extreme(self, t, q=0.37, absmax=False):
        """
        Compute the q-th quantile extreme value (assuming a Gaussian process).
        Only applicable for positive real-valued spectra.

        The extreme value, ``x``, is calculated according to:

        ``x = sigma * sqrt(2 * ln((t / tz) / ln(1 / q)))``

        where ``sigma`` is the standard deviation of the process, ``t`` is the duration
        of the process, and ``q`` is the quantile. Setting ``q=0.37`` yields the
        most probable maximum (MPM).

        Parameters
        ----------
        t : float
            Time/duration in seconds for which the of the process is observed.
        q : float or array-like
            Quantile or sequence of quantiles to compute. Must be between 0 and 1
            (inclusive).
        absmax : bool
            Whether to compute absolute value extremes (i.e., taking the minima into account).
            If ``False`` (default), only the maxima are considered. See Notes.

        Returns
        -------
        x : float or array
            Extreme value(s). During the given time period, the maximum value (or
            absolute value maximum) of the process amplitudes will be below the
            returned value with the given probability.

        Notes
        -----
        Computing absolute value extremes by setting ``absmax=True`` is equivalent
        to doubling the expected zero-crossing rate, ``fz = 1 / Tz``.

        Notes
        -----
        The extreme values are calculated according to Equation (1.46) in reference [1]_.

        References
        ----------
        .. [1] A. Naess and T. Moan, (2013), "Stochastic dynamics of marine structures",
           Cambridge University Press.

        """

        if absmax:
            tz = self.tz / 2.0
        else:
            tz = self.tz

        q = np.asarray_chkfinite(q)
        return self.std() * np.sqrt(2.0 * np.log((t / tz) / np.log(1.0 / q)))


class DirectionalSpectrum(_SpectrumMixin, Grid):
    """
    Directional spectrum.

    The ``DirectionalSpectrum`` class extends the :class:`~waveresponse.Grid`
    class, and is a two-dimentional frequency/(wave)direction grid. The spectrum values
    represents spectrum density.

    Proper scaling is applied to ensure that the total "energy" remains constant at all times.

    Parameters
    ----------
    freq : array-like
        1-D array of grid frequency coordinates. Positive and monotonically increasing.
    dirs : array-like
        1-D array of grid direction coordinates. Positive and monotonically increasing.
        Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
    vals : array-like (N, M)
        Spectrum density values associated with the grid. Should be a 2-D array
        of shape (N, M), such that ``N=len(freq)`` and ``M=len(dirs)``.
    freq_hz : bool
        If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
    degrees : bool
        If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
    clockwise : bool
        If positive directions are defined to be 'clockwise' (``True``) or 'counterclockwise'
        (``False``). Clockwise means that the directions follow the right-hand rule
        with an axis pointing downwards.
    waves_coming_from : bool
        If waves are 'coming from' the given directions. If ``False``, 'going towards'
        convention is assumed.
    """

    def __init__(
        self,
        freq,
        dirs,
        vals,
        freq_hz=False,
        degrees=False,
        clockwise=False,
        waves_coming_from=True,
    ):
        super().__init__(
            freq,
            dirs,
            vals,
            freq_hz=freq_hz,
            degrees=degrees,
            clockwise=clockwise,
            waves_coming_from=waves_coming_from,
        )

        if freq_hz:
            self._vals = 1.0 / (2.0 * np.pi) * self._vals

        if degrees:
            self._vals = 180.0 / np.pi * self._vals

    def __repr__(self):
        return "DirectionalSpectrum"

    def _freq_spectrum(self, freq_hz=None):
        """
        Integrate the spectrum over the directional domain to obtain the non-directional
        spectrum.

        Parameters
        ----------
        freq_hz : bool
            If frequencies should be returned in 'Hz'. If ``False``, 'rad/s' is
            used. Defaults to original unit used during initialization.

        Returns
        -------
        f : array
            1-D array of frequency coordinates in 'Hz' or 'rad/s'.
        s : array
            1-D array of spectral density values in 'm^2/Hz' or 'm^2/(rad/s)'.
        """

        if freq_hz is None:
            freq_hz = self._freq_hz

        f, d, _ = self.grid(freq_hz=freq_hz, degrees=False)
        d = self._full_range_dir(d)
        vv = self.interpolate(f, d, freq_hz=freq_hz, degrees=False)

        s = np.array([trapezoid(vv_d, d) for vv_d in vv])

        return f, s

    @staticmethod
    def _full_range_dir(x):
        """Add direction range bounds (0.0 and 2.0 * np.pi)"""
        range_end = np.nextafter(2.0 * np.pi, 0.0, dtype=type(x[0]))

        if x[0] != 0.0:
            x = np.r_[0.0, x]
        if x[-1] < range_end:
            x = np.r_[x, range_end]
        return x

    @classmethod
    def from_spectrum1d(
        cls,
        freq,
        dirs,
        spectrum1d,
        spread_fun,
        dirp,
        freq_hz=False,
        degrees=False,
        clockwise=False,
        waves_coming_from=True,
    ):
        """
        Construct a 2-D 'directional' spectrum from a 1-D 'non-directional' spectrum,
        a spreading function and a peak direction.

        The directional spectrum is constructed according to:

            ``S(f, theta) = S(f) * D(f, theta - theta_p)``

        where ``S(f)`` is the non-directional spectrum, ``D(f, theta - theta_p)``
        is the directional spreading function, and ``theta_p`` is the peak direction.
        ``f`` is the frequency coordinate, and ``theta`` is the direction coordinate.

        Parameters
        ----------
        freq : array-like
            1-D array of grid frequency coordinates. Positive and monotonically increasing.
        dirs : array-like
            1-D array of grid direction coordinates. Positive and monotonically increasing.
            Must cover the directional range [0, 360) degrees (or [0, 2 * pi) radians).
        spectrum1d : array-like
            1-D array of non-directional spectrum density values. These 1-D spectrum
            values will be scaled according to the spreading function, and distributed
            to all frequency/direction coordinates. `spectrum1d` must have the same
            length as `freq`.
        spread_fun : callable
            Spreading function. Takes a frequency coordinate (float) and a direction
            coordinate (float) as input, and returns a corresponding scaling value
            (float).
        dirp : float
            Peak direction. Direction in which the spectrum has its maximum values.
        freq_hz : bool
            If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
        degrees : bool
            If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
        clockwise : bool
            If positive directions are defined to be 'clockwise'. If ``False``,
            'counterclockwise' is assumed.
        waves_coming_from : bool
            If waves are 'coming from' the given directions. If ``False``, 'going towards'
            convention is assumed.
        """
        spectrum1d = np.asarray_chkfinite(spectrum1d).reshape(-1, 1)
        vals = np.tile(spectrum1d, (1, len(dirs)))

        if degrees:
            period = 360.0
        else:
            period = 2.0 * np.pi

        for (idx_f, idx_d), val_i in np.ndenumerate(vals):
            f_i = freq[idx_f]
            d_i = _robust_modulus(dirs[idx_d] - dirp, period)
            vals[idx_f, idx_d] = spread_fun(f_i, d_i) * val_i

        return cls(
            freq,
            dirs,
            vals,
            freq_hz=freq_hz,
            degrees=degrees,
            clockwise=clockwise,
            waves_coming_from=waves_coming_from,
        )

    def grid(self, freq_hz=False, degrees=False):
        """
        Return a copy of the spectrum's frequency/direction coordinates and corresponding
        values.

        Parameters
        ----------
        freq_hz : bool
            If frequencies should be returned in 'Hz'. If ``False``, 'rad/s' is used.
        degrees : bool
            If directions should be returned in 'degrees'. If ``False``, 'radians'
            is used.

        Returns
        -------
        freq : array
            1-D array of grid frequency coordinates.
        dirs : array
            1-D array of grid direction coordinates.
        vals : array (N, M)
            Spectrum density values as 2-D array of shape (N, M), such that ``N=len(freq)``
            and ``M=len(dirs)``.
        """
        freq, dirs, vals = super().grid(freq_hz=freq_hz, degrees=degrees)

        if freq_hz:
            vals *= 2.0 * np.pi

        if degrees:
            vals *= np.pi / 180.0

        return freq, dirs, vals

    def bingrid(self, freq_hz=False, degrees=False, complex_convert="rectangular"):
        """
        Return a copy of the spectrum's frequency and direction coordinates,
        along with the corresponding binned spectrum values.

        The spectrum values are interpolated and then integrated over their
        respective direction bins, resulting in total energy per unit frequency.
        This differs from the ``grid`` method, which returns the spectral density
        values directly without bin integration.

        Parameters
        ----------
        freq_hz : bool
            Whether to return frequencies in hertz (Hz). If ``False``, angular
            frequency in radians per second (rad/s) is used.
        degrees : bool
            Whether to return directions in degrees. If ``False``, radians are used.
        complex_convert : str, optional
            How to convert complex number grid values before interpolating. Should
            be 'rectangular' or 'polar'. If 'rectangular' (default), complex values
            are converted to rectangular form (i.e., real and imaginary part) before
            interpolating. If 'polar', the values are instead converted to polar
            form (i.e., amplitude and phase) before interpolating. The values are
            converted back to complex form after interpolation.

        Returns
        -------
        freq : ndarray, shape (N,)
            1D array of frequency coordinates.
        dirs : ndarray, shape (M,)
            1D array of direction coordinates.
        vals : ndarray, shape (N, M)
            2D array of binned spectrum energy values (energy per unit frequency).
            ``N = len(freq)``, ``M = len(dirs)``.
        """

        dirs_tmp = np.r_[
            -2.0 * np.pi + self._dirs[-1], self._dirs, 2.0 * np.pi + self._dirs[0]
        ]

        dirs_bin = np.empty(self._dirs.size * 2 + 1, dtype=self._dirs.dtype)
        dirs_bin[0::2] = dirs_tmp[:-1] + np.diff(dirs_tmp) / 2.0  # bin boundaries
        dirs_bin[1::2] = self._dirs  # bin centers

        interp_fun = _GridInterpolator(
            self._freq,
            self._dirs,
            self._vals,
            complex_convert=complex_convert,
            method="linear",
            bounds_error=True,
        )
        vals_tmp = interp_fun(self._freq, dirs_bin)

        vals_binned = np.column_stack(
            [
                trapezoid(vals_tmp[:, i : i + 3], dirs_bin[i : i + 3], axis=1)
                for i in range(0, 2 * len(self._dirs), 2)
            ]
        )

        if freq_hz:
            vals_binned *= 2.0 * np.pi

        return self.freq(freq_hz=freq_hz), self.dirs(degrees=degrees), vals_binned

    def interpolate(
        self,
        freq,
        dirs,
        freq_hz=False,
        degrees=False,
        complex_convert="rectangular",
        fill_value=0.0,
    ):
        """
        Interpolate (linear) the grid values to match the given frequency and direction
        coordinates.

        A 'fill value' is used for extrapolation (i.e. `freq` outside the bounds
        of the provided 2-D grid). Directions are treated as periodic.

        Parameters
        ----------
        freq : array-like
            1-D array of grid frequency coordinates. Positive and monotonically increasing.
        dirs : array-like
            1-D array of grid direction coordinates. Positive and monotonically increasing.
        freq_hz : bool
            If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
        degrees : bool
            If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
        complex_convert : str, optional
            How to convert complex number grid values before interpolating. Should
            be 'rectangular' or 'polar'. If 'rectangular' (default), complex values
            are converted to rectangular form (i.e., real and imaginary part) before
            interpolating. If 'polar', the values are instead converted to polar
            form (i.e., amplitude and phase) before interpolating. The values are
            converted back to complex form after interpolation.
        fill_value : float or None
            The value used for extrapolation (i.e., `freq` outside the bounds of
            the provided grid). If ``None``, values outside the frequency domain
            are extrapolated via nearest-neighbor extrapolation. Note that directions
            are treated as periodic (and will not need extrapolation).

        Returns
        -------
        array :
            Interpolated grid values.
        """
        warnings.warn(
            "The `DirectionalSpectrum.interpolate` method is deprecated and will be removed in a future release."
            "Use `DirectionalSpectrum.reshape` method instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        vals = super().interpolate(
            freq,
            dirs,
            freq_hz=freq_hz,
            degrees=degrees,
            complex_convert=complex_convert,
            fill_value=fill_value,
        )

        if freq_hz:
            vals *= 2.0 * np.pi

        if degrees:
            vals *= np.pi / 180.0
        return vals

    def reshape(
        self,
        freq,
        dirs,
        freq_hz=False,
        degrees=False,
        complex_convert="rectangular",
        fill_value=0.0,
    ):
        """
        Reshape the grid to match the given frequency/direction coordinates. Grid
        values will be interpolated (linear).

        Parameters
        ----------
        freq : array-like
            1-D array of new grid frequency coordinates. Positive and monotonically
            increasing.
        dirs : array-like
            1-D array of new grid direction coordinates. Positive and monotonically increasing.
            Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
        freq_hz : bool
            If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
        degrees : bool
            If direction is given in 'degrees'. If ``False``, 'radians' are assumed.
        complex_convert : str, optional
            How to convert complex number grid values before interpolating. Should
            be 'rectangular' or 'polar'. If 'rectangular' (default), complex values
            are converted to rectangular form (i.e., real and imaginary part) before
            interpolating. If 'polar', the values are instead converted to polar
            form (i.e., amplitude and phase) before interpolating. The values are
            converted back to complex form after interpolation.
        fill_value : float or None
            The value used for extrapolation (i.e., `freq` outside the bounds of
            the provided grid). If ``None``, values outside the frequency domain
            are extrapolated via nearest-neighbor extrapolation. Note that directions
            are treated as periodic (and will not need extrapolation).

        Returns
        -------
        obj :
            A copy of the object where the underlying coordinate system is reshaped.
        """
        freq_new = np.asarray_chkfinite(freq).copy()
        dirs_new = np.asarray_chkfinite(dirs).copy()

        if freq_hz:
            freq_new = 2.0 * np.pi * freq_new

        if degrees:
            dirs_new = (np.pi / 180.0) * dirs_new

        self._check_freq(freq_new)
        self._check_dirs(dirs_new)

        interp_fun = _GridInterpolator(
            self._freq,
            self._dirs,
            self._vals,
            complex_convert=complex_convert,
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

        vals_new = interp_fun(freq_new, dirs_new)

        new = self.copy()
        new._freq, new._dirs, new._vals = freq_new, dirs_new, vals_new
        return new


class DirectionalBinSpectrum(_SpectrumMixin, Grid):
    """
    Directional binned spectrum.

    The ``DirectionalBinSpectrum`` class extends the :class:`~waveresponse.Grid`
    class and represents a two-dimensional frequency/wave-direction grid. The spectrum values
    represent spectral density as a function of frequency, binned by direction.

    Proper scaling is applied to ensure that the total "energy" remains constant at all times.

    Parameters
    ----------
    freq : array-like
        1-D array of grid frequency coordinates. Positive and monotonically increasing.
    dirs : array-like
        1-D array of grid direction coordinates. Positive and monotonically increasing.
        Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
    vals : array-like (N, M)
        Spectrum density values associated with the grid. Should be a 2-D array
        of shape (N, M), such that ``N=len(freq)`` and ``M=len(dirs)``.
    freq_hz : bool
        If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
    degrees : bool
        If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
    clockwise : bool
        If positive directions are defined to be 'clockwise' (``True``) or 'counterclockwise'
        (``False``). Clockwise means that the directions follow the right-hand rule
        with an axis pointing downwards.
    waves_coming_from : bool
        If waves are 'coming from' the given directions. If ``False``, 'going towards'
        convention is assumed.
    """

    def __init__(
        self,
        freq,
        dirs,
        vals,
        freq_hz=False,
        degrees=False,
        clockwise=False,
        waves_coming_from=True,
    ):
        super().__init__(
            freq,
            dirs,
            vals,
            freq_hz=freq_hz,
            degrees=degrees,
            clockwise=clockwise,
            waves_coming_from=waves_coming_from,
        )

        if freq_hz:
            self._vals = 1.0 / (2.0 * np.pi) * self._vals

    def __repr__(self):
        return "DirectionalBinSpectrum"

    def _freq_spectrum(self, freq_hz=None):
        """
        Integrate the spectrum over the directional domain to obtain the non-directional
        spectrum.

        Parameters
        ----------
        freq_hz : bool
            If frequencies should be returned in 'Hz'. If ``False``, 'rad/s' is
            used. Defaults to original unit used during initialization.

        Returns
        -------
        f : array
            1-D array of frequency coordinates in 'Hz' or 'rad/s'.
        s : array
            1-D array of spectral density values in 'm^2/Hz' or 'm^2/(rad/s)'.
        """

        if freq_hz is None:
            freq_hz = self._freq_hz

        f, _, vv = self.grid(freq_hz=freq_hz, degrees=False)
        s = vv.sum(axis=1)

        return f, s

    def grid(self, freq_hz=False, degrees=False):
        """
        Return a copy of the spectrum's frequency/direction coordinates and corresponding
        values.

        Parameters
        ----------
        freq_hz : bool
            If frequencies should be returned in 'Hz'. If ``False``, 'rad/s' is used.
        degrees : bool
            If directions should be returned in 'degrees'. If ``False``, 'radians'
            is used.

        Returns
        -------
        freq : array
            1-D array of grid frequency coordinates.
        dirs : array
            1-D array of grid direction coordinates.
        vals : array (N, M)
            Spectrum density values as 2-D array of shape (N, M), such that ``N=len(freq)``
            and ``M=len(dirs)``.
        """
        freq, dirs, vals = super().grid(freq_hz=freq_hz, degrees=degrees)

        if freq_hz:
            vals *= 2.0 * np.pi

        return freq, dirs, vals

    def interpolate(self, *args, **kwargs):
        raise AttributeError("Use `.reshape` instead.")

    def reshape(
        self,
        freq,
        freq_hz=False,
        complex_convert="rectangular",
        fill_value=0.0,
    ):
        """
        Reshape the grid to match the given frequency coordinates. Grid
        values will be interpolated (linear).

        Parameters
        ----------
        freq : array-like
            1-D array of new grid frequency coordinates. Positive and monotonically
            increasing.
        freq_hz : bool
            If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
        complex_convert : str, optional
            How to convert complex number grid values before interpolating. Should
            be 'rectangular' or 'polar'. If 'rectangular' (default), complex values
            are converted to rectangular form (i.e., real and imaginary part) before
            interpolating. If 'polar', the values are instead converted to polar
            form (i.e., amplitude and phase) before interpolating. The values are
            converted back to complex form after interpolation.
        fill_value : float or None
            The value used for extrapolation (i.e., `freq` outside the bounds of
            the provided grid). If ``None``, values outside the frequency domain
            are extrapolated via nearest-neighbor extrapolation. Note that directions
            are treated as periodic (and will not need extrapolation).

        Returns
        -------
        obj :
            A copy of the object where the underlying coordinate system is reshaped.
        """
        freq_new = np.asarray_chkfinite(freq).copy()

        if freq_hz:
            freq_new = 2.0 * np.pi * freq_new

        self._check_freq(freq_new)

        interp_fun = _GridInterpolator(
            self._freq,
            self._dirs,
            self._vals,
            complex_convert=complex_convert,
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

        vals_new = interp_fun(freq_new, self._dirs)

        new = self.copy()
        new._freq, new._vals = freq_new, vals_new
        return new


class WaveSpectrum(DisableComplexMixin, DirectionalSpectrum):
    """
    Wave spectrum.

    The ``WaveSpectrum`` class extends the :class:`~waveresponse.DirectionalSpectrum`
    class, and is a two-dimentional frequency/(wave)direction grid. The spectrum values
    represents spectrum density. Only real and positive values allowed.

    Proper scaling is applied to ensure that the total "energy" remains constant at all times.

    Parameters
    ----------
    freq : array-like
        1-D array of grid frequency coordinates. Positive and monotonically increasing.
    dirs : array-like
        1-D array of grid direction coordinates. Positive and monotonically increasing.
        Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
    vals : array-like (N, M)
        Spectrum density values associated with the grid. Should be a 2-D array
        of shape (N, M), such that ``N=len(freq)`` and ``M=len(dirs)``.
    freq_hz : bool
        If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
    degrees : bool
        If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
    clockwise : bool
        If positive directions are defined to be 'clockwise' (``True``) or 'counterclockwise'
        (``False``). Clockwise means that the directions follow the right-hand rule
        with an axis pointing downwards.
    waves_coming_from : bool
        If waves are 'coming from' the given directions. If ``False``, 'going towards'
        convention is assumed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if np.any(np.iscomplex(self._vals)):
            raise ValueError("Spectrum values can not be complex.")
        elif np.any(self._vals < 0.0):
            raise ValueError("Spectrum values must be positive.")

    def __repr__(self):
        return "WaveSpectrum"

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

    @property
    def tp(self):
        """
        Wave peak period in 'seconds'.

        The period at which the 'non-directional' wave spectrum, ``S(f)``, has its maximum
        value.
        """
        f, S = self.spectrum1d(axis=1, freq_hz=True)
        fp = f[np.argmax(S)]
        return 1.0 / fp

    @staticmethod
    def _mean_direction(dirs, spectrum):
        """
        Mean spectrum direction.

        Parameters
        ----------
        dirs : array-like
            Directions in 'radians'.
        spectrum : array-like
            1-D spectrum directional distribution.
        """
        sin = trapezoid(np.sin(dirs) * spectrum, dirs)
        cos = trapezoid(np.cos(dirs) * spectrum, dirs)
        return _robust_modulus(np.arctan2(sin, cos), 2.0 * np.pi)

    def dirp(self, degrees=None):
        """
        Wave peak direction.

        Defined as the mean wave direction along the frequency corresponding to
        the maximum value of the 'non-directional' spectrum.

        Parameters
        ----------
        degrees : bool
            If wave peak direction should be returned in 'degrees'. If ``False``,
            the direction is returned in 'radians'. Defaults to original unit used
            during initialization.
        """

        if degrees is None:
            degrees = self._degrees

        freq, spectrum1d = self.spectrum1d(axis=1, freq_hz=False)

        dirs = self._full_range_dir(self._dirs)  # radians
        spectrum2d = self.interpolate(freq, dirs, freq_hz=False, degrees=False)

        spectrum_peak_dir = spectrum2d[np.argmax(spectrum1d), :]

        dirp = self._mean_direction(dirs, spectrum_peak_dir)

        if degrees:
            dirp = (180.0 / np.pi) * dirp

        return dirp

    def dirm(self, degrees=None):
        """
        Mean wave direction.

        Parameters
        ----------
        degrees : bool
            If mean wave direction should be returned in 'degrees'. If ``False``,
            the direction is returned in 'radians'. Defaults to original unit used
            during instantiation.
        """

        if degrees is None:
            degrees = self._degrees

        dp, sp = self.spectrum1d(axis=0, degrees=False)

        d = self._full_range_dir(dp)
        spectrum_dir = np.interp(d, dp, sp, period=2.0 * np.pi)

        dirm = self._mean_direction(d, spectrum_dir)

        if degrees:
            dirm = np.degrees(dirm)

        return dirm


class WaveBinSpectrum(DisableComplexMixin, DirectionalBinSpectrum):
    """
    Binned wave spectrum.

    The ``WaveSpectrum`` class extends the :class:`~waveresponse.DirectionalBinSpectrum`
    class, and is a two-dimentional frequency/(wave)direction grid. The spectrum values
    represents spectrum density as a function of frequency, binned by direction.

    Proper scaling is applied to ensure that the total "energy" remains constant at all times.

    Parameters
    ----------
    freq : array-like
        1-D array of grid frequency coordinates. Positive and monotonically increasing.
    dirs : array-like
        1-D array of grid direction coordinates. Positive and monotonically increasing.
        Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
    vals : array-like (N, M)
        Spectrum density values associated with the grid. Should be a 2-D array
        of shape (N, M), such that ``N=len(freq)`` and ``M=len(dirs)``.
    freq_hz : bool
        If frequency is given in 'Hz'. If ``False``, 'rad/s' is assumed.
    degrees : bool
        If direction is given in 'degrees'. If ``False``, 'radians' is assumed.
    clockwise : bool
        If positive directions are defined to be 'clockwise' (``True``) or 'counterclockwise'
        (``False``). Clockwise means that the directions follow the right-hand rule
        with an axis pointing downwards.
    waves_coming_from : bool
        If waves are 'coming from' the given directions. If ``False``, 'going towards'
        convention is assumed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if np.any(np.iscomplex(self._vals)):
            raise ValueError("Spectrum values can not be complex.")
        elif np.any(self._vals < 0.0):
            raise ValueError("Spectrum values must be positive.")

    def __repr__(self):
        return "WaveBinSpectrum"

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

    @property
    def tp(self):
        """
        Wave peak period in 'seconds'.

        The period at which the 'non-directional' wave spectrum, ``S(f)``, has its maximum
        value.
        """
        f, S = self.spectrum1d(axis=1, freq_hz=True)
        fp = f[np.argmax(S)]
        return 1.0 / fp

    @staticmethod
    def _mean_direction(dirs, spectrum):
        """
        Mean spectrum direction.

        Parameters
        ----------
        dirs : array-like
            Directions in 'radians'.
        spectrum : array-like
            1-D spectrum directional distribution.
        """
        sin = np.sum(np.sin(dirs) * spectrum) / len(dirs)
        cos = np.sum(np.cos(dirs) * spectrum) / len(dirs)
        return _robust_modulus(np.arctan2(sin, cos), 2.0 * np.pi)

    def dirp(self, degrees=None):
        """
        Wave peak direction.

        Defined as the mean wave direction along the frequency corresponding to
        the maximum value of the 'non-directional' spectrum.

        Parameters
        ----------
        degrees : bool
            If wave peak direction should be returned in 'degrees'. If ``False``,
            the direction is returned in 'radians'. Defaults to original unit used
            during initialization.
        """

        if degrees is None:
            degrees = self._degrees

        _, spectrum1d = self.spectrum1d(axis=1, freq_hz=False)

        spectrum_peakfreq = self._vals[np.argmax(spectrum1d), :]

        dirp = self._mean_direction(self._dirs, spectrum_peakfreq)

        if degrees:
            dirp = (180.0 / np.pi) * dirp

        return dirp

    def dirm(self, degrees=None):
        """
        Mean wave direction.

        Parameters
        ----------
        degrees : bool
            If mean wave direction should be returned in 'degrees'. If ``False``,
            the direction is returned in 'radians'. Defaults to original unit used
            during instantiation.
        """

        if degrees is None:
            degrees = self._degrees

        dirm = self._mean_direction(*self.spectrum1d(axis=0, degrees=False))

        if degrees:
            dirm = np.degrees(dirm)

        return dirm


def _calculate_response_deprecated(rao, wave_body, coord_freq, coord_dirs):
    # TODO: Deprecated. Remove this function in a future release.

    if coord_freq == "wave":
        freq = wave_body._freq
    elif coord_freq == "rao":
        freq = rao._freq
    else:
        raise ValueError("Invalid `coord_freq` value. Should be 'wave' or 'rao'.")
    if coord_dirs == "wave":
        dirs = wave_body._dirs
    elif coord_dirs == "rao":
        dirs = rao._dirs
    else:
        raise ValueError("Invalid `coord_dirs` value. Should be 'wave' or 'rao'.")

    rao_squared = (rao * rao.conjugate()).real
    rao_squared = rao_squared.reshape(freq, dirs, freq_hz=False, degrees=False)
    wave_body = wave_body.reshape(freq, dirs, freq_hz=False, degrees=False)

    return multiply(rao_squared, wave_body, output_type="DirectionalSpectrum")


def calculate_response(
    rao,
    wave,
    heading,
    heading_degrees=False,
    reshape="rao_squared",
    coord_freq=None,
    coord_dirs=None,
):
    """
    Calculate response spectrum.

    The response spectrum is calculated according to:

        S_x(w, theta) = H(w, theta) * H*(w, theta) * S_w(w, theta)

    where S_x(w, theta) denotes the response spectrum, H(w, theta) denotes the RAO,
    H*(w, theta) denotes the RAO conjugate, and S_w(w, theta) denotes the wave
    spectrum (expressed in the RAO's reference frame).

    The frequency and direction coordinates are dictatated by the wave spectrum.
    I.e., the RAO (or the magnitude-squared verison of it) is interpolated to match
    the grid coordinates of the wave spectrum.

    Parameters
    ----------
    rao : RAO
        Response amplitude operator (RAO).
    wave : WaveSpectrum or WaveBinSpectrum
        2-D wave spectrum.
    heading : float
        Heading of vessel relative to wave spectrum coordinate system.
    heading_degrees : bool
        Whether the heading is given in 'degrees'. If ``False``, 'radians' is assumed.
    reshape : {'rao', 'rao_squared'}, default 'rao_squared'
        Determines whether to reshape the RAO or the magnitude-squared version of
        the RAO before pairing with the wave spectrum. Linear interpolation is
        performed to match the frequency and direction coordinates of the wave
        spectrum.
    coord_freq : str, optional
        Deprecated; use `reshape` instead. Frequency coordinates for interpolation.
        Should be 'wave' or 'rao'. Determines if it is the wave spectrum or the
        RAO that should dictate which frequencies to use in response calculation.
        The other object will be interpolated to match these frequencies.
    coord_dirs : str, optional
        Deprecated; use `reshape` instead. Direction coordinates for interpolation.
        Should be 'wave' or 'rao'. Determines if it is the wave spectrum or the
        RAO that should dictate which directions to use in response calculation.
        The other object will be interpolated to match these directions.

    Returns
    -------
    DirectionalSpectrum or DirectionalBinSpectrum :
        Response spectrum.
    """
    wave_body = wave.rotate(heading, degrees=heading_degrees)
    wave_body.set_wave_convention(**rao.wave_convention)

    # TODO: Remove once the deprecation period is over
    if coord_freq and coord_dirs:
        warnings.warn(
            "The `coord_freq` and `coord_dirs` parameters are deprecated and will be removed in a future release."
            "Use the `reshape` parameter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _calculate_response_deprecated(rao, wave_body, coord_freq, coord_dirs)
    elif coord_freq or coord_dirs:
        raise ValueError("Both `coord_freq` and `coord_dirs` must be provided.")

    freq, dirs = wave_body._freq, wave_body._dirs
    if reshape == "rao":
        rao = rao.reshape(freq, dirs, freq_hz=False, degrees=False)
        rao_squared = (rao * rao.conjugate()).real
    elif reshape == "rao_squared":
        rao_squared = (rao * rao.conjugate()).real
        rao_squared = rao_squared.reshape(freq, dirs, freq_hz=False, degrees=False)
    else:
        raise ValueError("Invalid `reshape` value. Should be 'rao' or 'rao_squared'.")

    TYPE_MAP = {
        WaveSpectrum: "DirectionalSpectrum",
        WaveBinSpectrum: "DirectionalBinSpectrum",
    }

    try:
        type_ = TYPE_MAP[type(wave)]
    except KeyError:
        raise ValueError(
            "Invalid `wave` type. Should be 'WaveSpectrum' or 'WaveBinSpectrum'."
        )

    return multiply(rao_squared, wave_body, output_type=type_)


class BaseSpreading(ABC):
    """
    Base class for spreading functions.

    Parameters
    ----------
    freq_hz : bool
        If frequencies passed to the spreading function will be given in 'Hz'.
        If ``False``, 'rad/s' is assumed.
    degrees : bool
        If directions passed to the spreading function will be given in 'degrees'.
        If ``False``, 'radians' is assumed.
    """

    def __init__(self, freq_hz=False, degrees=False):
        self._freq_hz = freq_hz
        self._degrees = degrees

    def __call__(self, frequency, direction):
        """
        Get spreading value for given frequency/direction coordinate.

        Parameters
        ----------
        frequency : float
            Frequency coordinate. Units should be according to the `freq_hz` flag
            given during initialization.
        direction : float
            Direction coordinate. Units should be according to the `degrees` flag
            given during initialization.
        """
        if self._freq_hz:
            frequency = 2.0 * np.pi * frequency

        if self._degrees:
            direction = (np.pi / 180.0) * direction
            scale = 1.0 / 360.0
        else:
            scale = 1.0 / (2.0 * np.pi)

        return scale * self._spread_fun(
            frequency, _robust_modulus(direction, 2.0 * np.pi)
        )

    @abstractmethod
    def _spread_fun(self, omega, theta):
        """
        Get spreading value for given frequency/direction coordinate.

        Parameters
        ----------
        omega : float
            Frequency coordinate in 'rad/s'.
        direction : float
            Direction coordinate in 'radians'.
        """
        raise NotImplementedError()

    def discrete_directions(self, n, direction_offset=0.0):
        """
        Split the spreading function into discrete direction bins with
        "equal energy", i.e. equal area under the curve. The direcitons
        representing the bins are chosen to have equal area under the curve on
        each side within the bin.

        Parameters
        ----------
        n : int
            Number of discrete directions.
        direction_offset : float, default
            A offset to add to the discrete directions. Units should be
            according to the `degrees` flag given during initialization.

        Returns
        -------
        ndarray
            A sequence of direction representing "equal energy" bins with range
            wrapped to [0, 360) degrees or [0, 2 * pi) radians according
            to the `degrees` flag given during initialization.
        """
        if self._degrees:
            x_lb = -180.0
            x_ub = 180.0
            periodicity = 360.0
        else:
            x_lb = -np.pi
            x_ub = np.pi
            periodicity = 2.0 * np.pi

        total_area, _ = quad(
            lambda theta: self(None, theta), x_lb, x_ub, epsabs=1.0e-6, epsrel=0.0
        )

        half_bin_edges = np.empty(2 * n - 1)

        x_prev = x_lb
        for i in range(1, 2 * n):
            target_area = total_area * i / (2 * n)
            res = root_scalar(
                lambda x: quad(
                    lambda theta: self(None, theta), x_lb, x, epsabs=1.0e-6, epsrel=0.0
                )[0]
                - target_area,
                bracket=[x_prev, x_ub],
            )

            if not res.converged:
                raise RuntimeError(f"Failed find the directions: {res.flag}")

            x_prev = res.root
            half_bin_edges[i - 1] = x_prev

        return _robust_modulus(half_bin_edges[::2] + direction_offset, periodicity)


class CosineHalfSpreading(BaseSpreading):
    """
    Cosine-2s type spreading (half directional range).

    Defined as:

        ``D(theta) = scale * C(s) * cos(theta) ** (2 * s)`` , for -pi/2 <= theta <= pi/2

        ``D(theta) = 0`` , otherwise

    where,

        ``C(s) = 2 ** (2 * s + 1) * gamma(s + 1) ** 2 / gamma(2 * s + 1)``

    If `theta` is given in 'radians':

        ``scale = 1.0 / (2.0 * np.pi)``

    If `theta` is given in 'degrees':

        ``scale = 1 / 360.0``

    Note that this spreading is independent of frequency.

    Parameters
    ----------
    s : int
        Spreading coefficient.
    degrees : bool
        If directions passed to the spreading function will be given in 'degrees'.
        If ``False``, 'radians' is assumed.

    Notes
    -----
    The spreading function is implemented according to reference [1]_.

    References
    ----------
    .. [1] U. S. Army Engineer Waterways Experiment Station, Coastal Engineering Research
       Center. (1985, June). "Directional wave spectra using cosine-squared and cosine 2s
       spreading functions". Retrieved January 31, 2023, from
       https://apps.dtic.mil/sti/pdfs/ADA591687.pdf.
    """

    def __init__(self, s=1, degrees=False):
        self._s = s
        super().__init__(degrees=degrees)

    def _spread_fun(self, _, theta, /):
        if (np.pi / 2.0) <= theta <= (3.0 * np.pi / 2.0):
            return 0

        s = self._s
        c = 2 ** (2 * s + 1) * gamma(s + 1) ** 2 / gamma(2 * s + 1)
        return c * (np.cos(theta) ** 2.0) ** s


class CosineFullSpreading(BaseSpreading):
    """
    Cosine-2s type spreading (full directional range).

    Defined as:

        ``D(theta) = scale * C(s) * cos(theta / 2) ** (2 * s)``

    where,

        ``C(s) = 2 ** (2 * s) * gamma(s + 1) ** 2 / gamma(2 * s + 1)``

    If `theta` is given in 'radians':

        ``scale = 1.0 / (2.0 * np.pi)``

    If `theta` is given in 'degrees':

        ``scale = 1 / 360.0``

    Note that this spreading is independent of frequency.

    Parameters
    ----------
    s : int
        Spreading coefficient.
    degrees : bool
        If directions passed to the spreading function will be given in 'degrees'.
        If ``False``, 'radians' is assumed.

    Notes
    -----
    The spreading function is implemented according to reference [1]_.

    References
    ----------
    .. [1] U. S. Army Engineer Waterways Experiment Station, Coastal Engineering Research
       Center. (1985, June). "Directional wave spectra using cosine-squared and cosine 2s
       spreading functions". Retrieved January 31, 2023, from
       https://apps.dtic.mil/sti/pdfs/ADA591687.pdf.
    """

    def __init__(self, s=1, degrees=False):
        self._s = s
        super().__init__(degrees=degrees)

    def _spread_fun(self, _, theta, /):
        s = self._s
        c = 2 ** (2 * s) * gamma(s + 1) ** 2 / gamma(2 * s + 1)
        return c * (np.cos(theta / 2.0) ** 2.0) ** s
