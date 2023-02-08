import copy
from abc import ABC, abstractmethod
from numbers import Number

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp2d
from scipy.special import gamma


def _robust_modulus(x, periodicity):
    """
    Robust modulus operator.

    Similar to ``x % periodicity``, but ensures that it is robust w.r.t. floating
    point numbers.
    """
    x = np.asarray_chkfinite(x % periodicity).copy()
    return np.nextafter(x, -1, where=(x == periodicity), out=x)


def complex_to_polar(complex_vals, phase_degrees=False):
    """
    Convert complex numbers to polar form (i.e., amplitude and phase).

    Parameters
    ----------
    complex_vals : array-like
        Complex number values.
    phase_degrees : bool
        Whether the phase angles should be returned in 'degrees' (``True``) or
        'radians' (``False``).

    Returns
    -------
    amp : array
        Amplitudes.
    phase : array
        Phase angles.

    """
    complex_vals = np.asarray_chkfinite(complex_vals)
    amp = np.abs(complex_vals)
    phase = np.angle(complex_vals, deg=phase_degrees)
    return amp, phase


def polar_to_complex(amp, phase, phase_degrees=False):
    """
    Convert polar coordinates (i.e., amplitude and phase) to complex numbers.

    Parameters
    ----------
    complex_vals : array-like
        Complex number values.
    phase_degrees : bool
        Whether the phase angles are given in 'degrees'. If ``False``, 'radians'
        is assumed.

    Returns
    -------
    array :
        Complex numbers.
    """
    amp = np.asarray_chkfinite(amp)
    phase = np.asarray_chkfinite(phase)

    if phase_degrees:
        phase = (np.pi / 180.0) * phase

    if amp.shape != phase.shape:
        raise ValueError()

    return amp * (np.cos(phase) + 1j * np.sin(phase))


def _check_is_similar(*grids, exact_type=True):
    """
    Check if grid objects are similar.
    """
    grids = list(grids)
    grid_ref = grids.pop()

    if exact_type:
        type_ = type(grid_ref)

        def check_type(grid, grid_type):
            return type(grid) == grid_type

    else:
        type_ = Grid
        check_type = isinstance

    for grid_i in grids:
        if not check_type(grid_i, type_):
            raise TypeError("Object types are not similar.")
        elif grid_ref._vals.shape != grid_i._vals.shape:
            raise ValueError("Grid objects have different shape.")
        elif np.any(grid_ref._freq != grid_i._freq) or np.any(
            grid_ref._dirs != grid_i._dirs
        ):
            raise ValueError(
                "Grid objects have different frequency/direction coordinates."
            )
        elif grid_ref.wave_convention != grid_i.wave_convention:
            raise ValueError("Grid objects have different wave conventions.")


def multiply(grid1, grid2, output_type="grid"):
    """
    Multiply values (element-wise).

    Parameters
    ----------
    grid1 : obj
        Grid object.
    grid2 : obj
        Grid object.
    output_type : str {"grid", "rao", "directional_spectrum", "wave_spectrum"}
        Output grid type.
    """

    TYPE_MAP = {
        "grid": Grid,
        "rao": RAO,
        "directional_spectrum": DirectionalSpectrum,
        "wave_spectrum": WaveSpectrum,
    }

    if output_type not in TYPE_MAP:
        raise ValueError("The given `output_type` is not valid.")

    _check_is_similar(grid1, grid2, exact_type=False)

    type_ = TYPE_MAP.get(output_type)

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

    return type_.from_grid(new)


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
        ((dirs >= ub_1) | (dirs <= lb_1))
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
        If positive directions are defined to be 'clockwise'. If ``False``, 'counterclockwise'
        is assumed.
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

    @classmethod
    def from_grid(cls, grid):
        """
        Construct from a Grid object.

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
        if np.any(freq[:-1] >= freq[1:]) or freq[0] < 0:
            raise ValueError(
                "Frequencies must be positive and monotonically increasing."
            )

    def _check_dirs(self, dirs):
        """
        Check direction bins.
        """
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
            If positive directions are defined to be 'clockwise'. If ``False``,
            'counterclockwise' is assumed.
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
        elif complex_convert.lower() == "rectangular":
            interp_real = interp2d(xp, yp, np.real(zp), **kw)
            interp_imag = interp2d(xp, yp, np.imag(zp), **kw)
            return lambda *args, **kwargs: (
                interp_real(*args, **kwargs) + 1j * interp_imag(*args, **kwargs)
            )
        else:
            raise ValueError("Unknown 'complex_convert' type")

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

        Notes
        -----
        Apply 'polar' interpolation with caution as phase values are not "unwraped"
        before interpolation. This may lead to some unexpected artifacts in the
        results.

        """
        freq = np.asarray_chkfinite(freq).reshape(-1)
        dirs = np.asarray_chkfinite(dirs).reshape(-1)

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

        vals_new = self.interpolate(
            freq_new,
            dirs_new,
            freq_hz=False,
            degrees=False,
            complex_convert=complex_convert,
            fill_value=fill_value,
        )
        new = self.copy()
        new._freq, new._dirs, new._vals = freq_new, dirs_new, vals_new
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

    def __repr__(self):
        return "Grid"

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
        If positive directions are defined to be 'clockwise'. If ``False``, 'counterclockwise'
        is assumed.
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

    @classmethod
    def from_amp_phase(
        cls,
        freq,
        dirs,
        amp,
        phase,
        phase_degrees=False,
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
        return rao

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

    def to_amp_phase(self, phase_degrees=None, freq_hz=None, degrees=None):
        """
        Return the RAO as amplitude and phase values.

        Parameters
        ----------
        phase_degrees : bool
            If phase values should be returned in 'degrees'. If ``False``, 'radians'
            is used. Defaults to original units used during initialization or ``False``.
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

        freq, dirs, vals = self.grid(freq_hz=freq_hz, degrees=degrees)
        vals_amp, vals_phase = complex_to_polar(vals, phase_degrees=phase_degrees)
        return freq, dirs, vals_amp, vals_phase

    def __repr__(self):
        return "RAO"


class DirectionalSpectrum(DisableComplexMixin, Grid):
    """
    Directional spectrum.

    The ``DirectionalSpectrum`` class extends the :class:`~waveresponse.Grid`
    class, and is a two-dimentional frequency/(wave)direction grid. The spectrum values
    represents spectrum density.

    Proper scaling is performed such that the total "energy" is kept constant at
    all times.

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
        If positive directions are defined to be 'clockwise'. If ``False``, 'counterclockwise'
        is assumed.
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

        if np.any(np.iscomplex(self._vals)):
            raise ValueError("Spectrum values can not be complex.")
        elif np.any(self._vals < 0.0):
            raise ValueError("Spectrum values must be positive.")

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
            Must cover the directional range [0, 360) degrees (or [0, 2 * numpy.pi) radians).
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

    def __repr__(self):
        return "DirectionalSpectrum"

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

    def interpolate(
        self,
        freq,
        dirs,
        freq_hz=False,
        degrees=False,
        fill_value=0.0,
        **kwargs,
    ):
        """
        Interpolate (linear) the spectrum values to match the given frequency and direction
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
        fill_value : float or None
            The value used for extrapolation (i.e., `freq` outside the bounds of
            the provided grid). If ``None``, values outside the frequency domain
            are extrapolated via nearest-neighbor extrapolation. Note that directions
            are treated as periodic (and will not need extrapolation).

        Returns
        -------
        array :
            Interpolated spectrum density values.
        """

        vals = super().interpolate(
            freq,
            dirs,
            freq_hz=freq_hz,
            degrees=degrees,
            fill_value=fill_value,
        )

        if freq_hz:
            vals *= 2.0 * np.pi

        if degrees:
            vals *= np.pi / 180.0

        return vals

    @staticmethod
    def _full_range_dir(x):
        """Add direction range bounds (0.0 and 2.0 * np.pi)"""
        range_end = np.nextafter(2.0 * np.pi, 0.0, dtype=type(x[0]))

        if x[0] != 0.0:
            x = np.r_[0.0, x]
        if x[-1] < range_end:
            x = np.r_[x, range_end]
        return x

    def var(self):
        """
        Variance (integral) of the spectrum.
        """
        x = self._full_range_dir(self._dirs)
        y = self._freq
        zz = self.interpolate(y, x, freq_hz=False, degrees=False)
        return trapz([trapz(zz_x, x) for zz_x in zz], y)

    def std(self):
        """
        Standard deviation of the spectrum.
        """
        return np.sqrt(self.var())

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
        spectrum : 1-D array
            Spectrum density values, where the spectrum is integrated over the
            specified axis.
        """

        if freq_hz is None:
            freq_hz = self._freq_hz

        if axis == 1:
            degrees = False
        elif degrees is None:
            degrees = self._degrees

        freq, dirs, vals = self.grid(freq_hz=freq_hz, degrees=degrees)

        if axis == 0:
            y = freq
            x = dirs
            zz = vals.T
        elif axis == 1:
            y = self._full_range_dir(dirs)
            x = freq
            zz = self.interpolate(x, y, freq_hz=freq_hz, degrees=degrees)
        else:
            raise ValueError("'axis' must be 0 or 1.")

        spectrum = np.array([trapz(zz_y, y) for zz_y in zz])
        return x, spectrum

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
        f, spectrum = self.spectrum1d(axis=1, freq_hz=freq_hz)
        m_n = trapz((f**n) * spectrum, f)
        return m_n

    @property
    def tz(self):
        """
        Mean zero-crossing period, Tz, in 'seconds'.

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
        m0 = self.moment(0, freq_hz=True)
        m2 = self.moment(2, freq_hz=True)
        return np.sqrt(m0 / m2)

    def extreme(self, t, q=0.37, absmax=False):
        """
        Compute the q-th quantile extreme value (assuming a Gaussian process).

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


class WaveSpectrum(DirectionalSpectrum):
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
        sin = trapz(np.sin(dirs) * spectrum, dirs)
        cos = trapz(np.cos(dirs) * spectrum, dirs)
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

        dp, sp = self.spectrum1d(axis=0, degrees=False)

        d = self._full_range_dir(dp)
        spectrum_dir = np.interp(d, dp, sp, period=2.0 * np.pi)

        dirm = self._mean_direction(d, spectrum_dir)

        if degrees:
            dirm = np.degrees(dirm)

        return dirm


def calculate_response(
    rao, wave, heading, heading_degrees=False, coord_freq="wave", coord_dirs="wave"
):
    """
    Calculate response spectrum.

    Parameters
    ----------
    rao : obj
        Response amplitude operator (RAO) as a :class:`~waveresponse.RAO` object.
    wave : obj
        2-D wave spectrum as a :class:`~waveresponse.WaveSpectrum` object.
    heading : float
        Heading of vessel relative to wave spectrum coordinate system.
    heading_degrees : bool
        Whether the heading is given in 'degrees'. If ``False``, 'radians' is assumed.
    coord_freq : str, optional
        Frequency coordinates for interpolation. Should be 'wave' or 'rao'. Determines
        if it is the wave spectrum or the RAO that should dictate which frequencies
        to use in response calculation. The other object will be interpolated to
        match these frequencies.
    coord_dirs : str, optional
        Direction coordinates for interpolation. Should be 'wave' or 'rao'. Determines
        if it is the wave spectrum or the RAO that should dictate which directions
        to use in response calculation. The other object will be interpolated to
        match these directions.

    Returns
    -------
    obj :
        Response spectrum as :class:`DirectionalSpectrum` object.
    """
    wave_body = wave.rotate(heading, degrees=heading_degrees)
    wave_body.set_wave_convention(**rao.wave_convention)

    if coord_freq.lower() == "wave":
        freq = wave_body._freq
    elif coord_freq.lower() == "rao":
        freq = rao._freq
    else:
        raise ValueError("Invalid `coord_freq` value. Should be 'wave' or 'rao'.")

    if coord_dirs.lower() == "wave":
        dirs = wave_body._dirs
    elif coord_dirs.lower() == "rao":
        dirs = rao._dirs
    else:
        raise ValueError("Invalid `coord_dirs` value. Should be 'wave' or 'rao'.")

    rao_squared = (rao * rao.conjugate()).real
    rao_squared = rao_squared.reshape(freq, dirs, freq_hz=False, degrees=False)
    wave_body = wave_body.reshape(freq, dirs, freq_hz=False, degrees=False)

    return multiply(rao_squared, wave_body, output_type="directional_spectrum")


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
