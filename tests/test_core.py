import numpy as np
import pytest

from scarlet_lithium import (
    RAO,
    Grid,
    complex_to_polar,
    polar_to_complex,
    DirectionalSpectrum,
)


@pytest.fixture
def grid():
    freq = np.linspace(0, 1.0, 10)
    dirs = np.linspace(0, 360.0, 15, endpoint=False)
    vals = np.random.random((10, 15))
    grid = Grid(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=True,
        waves_coming_from=True,
    )
    return grid


@pytest.fixture
def rao():
    freq = np.linspace(0, 1.0, 10)
    dirs = np.linspace(0, 360.0, 15, endpoint=False)
    vals_amp = np.random.random((10, 15))
    vals_phase = np.random.random((10, 15))
    rao = RAO.from_amp_phase(
        freq,
        dirs,
        vals_amp,
        vals_phase,
        phase_degrees=False,
        freq_hz=True,
        degrees=True,
        clockwise=True,
        waves_coming_from=True,
    )
    return rao


class Test_complex_to_polar:
    def test_deg(self):
        complex_vals = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        amp_out, phase_out = complex_to_polar(complex_vals, phase_degrees=True)

        amp_expect = np.array([1.0, 1.0, 1.0])
        phase_expect = np.array([0.0, 90.0, 180.0])

        np.testing.assert_array_almost_equal(amp_out, amp_expect)
        np.testing.assert_array_almost_equal(phase_out, phase_expect)

    def test_rad(self):
        complex_vals = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        amp_out, phase_out = complex_to_polar(complex_vals, phase_degrees=False)

        amp_expect = np.array([1.0, 1.0, 1.0])
        phase_expect = np.array([0.0, np.pi / 2, np.pi])

        np.testing.assert_array_almost_equal(amp_out, amp_expect)
        np.testing.assert_array_almost_equal(phase_out, phase_expect)


class Test_polar_to_complex:
    def test_deg(self):
        amp = np.array([1.0, 1.0, 1.0])
        phase = np.array([0.0, 90.0, 180.0])
        complex_out = polar_to_complex(amp, phase, phase_degrees=True)

        complex_expect = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])

        np.testing.assert_array_almost_equal(complex_out, complex_expect)

    def test_rad(self):
        amp = np.array([1.0, 1.0, 1.0])
        phase = np.array([0.0, np.pi / 2, np.pi])
        complex_out = polar_to_complex(amp, phase, phase_degrees=False)

        complex_expect = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])

        np.testing.assert_array_almost_equal(complex_out, complex_expect)


class Test_Grid:
    def test__init__(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.zeros((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        np.testing.assert_array_almost_equal(grid._freq, 2.0 * np.pi * freq)
        np.testing.assert_array_almost_equal(grid._dirs, (np.pi / 180.0) * dirs)
        np.testing.assert_array_almost_equal(grid._vals, vals)
        assert grid._clockwise is True
        assert grid._waves_coming_from is True
        assert grid._freq_hz is True
        assert grid._degrees is True

    def test__init__2(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 2.0 * np.pi, 15, endpoint=False)
        vals = np.zeros((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=False,
            degrees=False,
            clockwise=False,
            waves_coming_from=False,
        )

        np.testing.assert_array_almost_equal(grid._freq, freq)
        np.testing.assert_array_almost_equal(grid._dirs, dirs)
        np.testing.assert_array_almost_equal(grid._vals, vals)
        assert grid._clockwise is False
        assert grid._waves_coming_from is False
        assert grid._freq_hz is False
        assert grid._degrees is False

    def test__init__raises_duplicate_freq(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 1, 2])
            dirs = np.array([0, 1, 2])
            vals = np.zeros((4, 3))
            Grid(freq, dirs, vals)

    def test__init__raises_duplicate_dirs(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 2])
            dirs = np.array([0, 1, 1, 2])
            vals = np.zeros((3, 4))
            Grid(freq, dirs, vals)

    def test__init__raises_negative_freq(self):
        with pytest.raises(ValueError):
            freq = np.array([-1, 1, 2, 3])
            dirs = np.array([0, 1, 2])
            vals = np.zeros((4, 3))
            Grid(freq, dirs, vals)

    def test__init__raises_negative_dirs(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 2, 3])
            dirs = np.array([-1, 1, 2])
            vals = np.zeros((4, 3))
            Grid(freq, dirs, vals)

    def test__init__raises_freq_not_increasing(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 2, 1])
            dirs = np.array([0, 1, 2])
            vals = np.zeros((4, 3))
            Grid(freq, dirs, vals)

    def test__init__raises_dirs_not_increasing(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 2])
            dirs = np.array([0, 1, 2, 1])
            vals = np.zeros((3, 4))
            Grid(freq, dirs, vals)

    def test__init__raises_dirs_2pi(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 2])
            dirs = np.array([0, 1, 2, 2.0 * np.pi])
            vals = np.zeros((3, 4))
            Grid(freq, dirs, vals, degrees=False)

    def test__init__raises_dirs_greater_than_2pi(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 2])
            dirs = np.array([0, 1, 2, 3.0 * np.pi])
            vals = np.zeros((3, 4))
            Grid(freq, dirs, vals, degrees=False)

    def test__init__raises_vals_shape(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 2])
            dirs = np.array([0, 1, 2, 3])
            vals = np.zeros((3, 10))
            Grid(freq, dirs, vals)

    def test_wave_convention(self, grid):
        convention_expect = {"clockwise": True, "waves_coming_from": True}
        convention_out = grid.wave_convention
        assert convention_out == convention_expect

    def test__sort(self):
        dirs_unsorted = [1, 3, 2, 4]
        vals_unsorted = [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
        dirs_sorted_out, vals_sorted_out = Grid._sort(dirs_unsorted, vals_unsorted)

        dirs_sorted_expect = np.array([1, 2, 3, 4])
        vals_sorted_expect = np.array(
            [
                [1, 3, 2, 4],
                [1, 3, 2, 4],
            ]
        )

        np.testing.assert_array_almost_equal(dirs_sorted_out, dirs_sorted_expect)
        np.testing.assert_array_almost_equal(vals_sorted_out, vals_sorted_expect)

    def test__convert_dirs_radians(self):
        dirs_in = np.array([0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4, np.pi])
        config_org = {"clockwise": False, "waves_coming_from": True}
        config_new = {"clockwise": True, "waves_coming_from": False}
        dirs_out = Grid._convert_dirs(dirs_in, config_new, config_org, degrees=False)

        dirs_expect = np.array([np.pi, 3.0 * np.pi / 4, np.pi / 2, np.pi / 4, 0])

        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)

    def test__convert_dirs_degrees(self):
        dirs_in = np.array([0, 45.0, 90.0, 135.0, 180.0])
        config_org = {"clockwise": False, "waves_coming_from": True}
        config_new = {"clockwise": True, "waves_coming_from": False}
        dirs_out = Grid._convert_dirs(dirs_in, config_new, config_org, degrees=True)

        dirs_expect = np.array([180.0, 135.0, 90.0, 45.0, 0])

        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)

    def test__convert(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.zeros((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=True,
        )

        freq_in = np.array([0.0, 0.5, 1.0])
        dirs_in = np.array([0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4, np.pi])
        vals_in = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        )
        config_org = {"clockwise": False, "waves_coming_from": True}
        config_new = {"clockwise": True, "waves_coming_from": False}
        freq_out, dirs_out, vals_out = grid._convert(
            freq_in, dirs_in, vals_in, config_new, config_org
        )

        freq_expect = freq_in
        dirs_expect = np.array([0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4, np.pi])
        vals_expect = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_set_wave_convention(self):
        freq_in = np.array([0.0, 0.5, 1.0])
        dirs_in = np.array([0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4, np.pi])
        vals_in = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        )
        grid = Grid(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=False,
            clockwise=False,
            waves_coming_from=True,
        )

        grid.set_wave_convention(clockwise=True, waves_coming_from=False)

        freq_expect = (2.0 * np.pi) * freq_in
        dirs_expect = np.array([0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4, np.pi])
        vals_expect = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        )

        np.testing.assert_array_almost_equal(grid._freq, freq_expect)
        np.testing.assert_array_almost_equal(grid._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(grid._vals, vals_expect)
        assert grid._clockwise is True
        assert grid._waves_coming_from is False

    def test_copy(self, grid):
        grid_copy = grid.copy()
        assert grid is grid
        assert grid_copy is not grid
        np.testing.assert_array_almost_equal(grid_copy._freq, grid._freq)
        np.testing.assert_array_almost_equal(grid_copy._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(grid_copy._vals, grid._vals)
        assert grid_copy._clockwise == grid._clockwise
        assert grid_copy._waves_coming_from == grid._waves_coming_from

    def test_rotate_deg(self):
        freq = np.array([0, 1])
        dirs = np.array([0, 90, 180])
        vals = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=True,
        )

        grid_rot = grid.rotate(45, degrees=True)

        freq_expect = (2.0 * np.pi) * np.array([0, 1])
        dirs_expect = (np.pi / 180.0) * np.array([45, 135, 315])
        vals_expect = np.array(
            [
                [2, 3, 1],
                [2, 3, 1],
            ]
        )

        np.testing.assert_array_almost_equal(grid_rot._freq, freq_expect)
        np.testing.assert_array_almost_equal(grid_rot._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(grid_rot._vals, vals_expect)

    def test_rotate_deg_neg(self):
        freq = np.array([0, 1])
        dirs = np.array([0, 90, 180])
        vals = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=True,
        )

        grid_rot = grid.rotate(-45, degrees=True)

        freq_expect = (2.0 * np.pi) * np.array([0, 1])
        dirs_expect = (np.pi / 180.0) * np.array([45, 135, 225])
        vals_expect = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )

        np.testing.assert_array_almost_equal(grid_rot._freq, freq_expect)
        np.testing.assert_array_almost_equal(grid_rot._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(grid_rot._vals, vals_expect)

    def test_rotate_rad(self):
        freq = np.array([0, 1])
        dirs = np.array([0, 90, 180])
        vals = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=True,
        )

        grid_rot = grid.rotate(np.pi / 4, degrees=False)

        freq_expect = (2.0 * np.pi) * np.array([0, 1])
        dirs_expect = (np.pi / 180.0) * np.array([45, 135, 315])
        vals_expect = np.array(
            [
                [2, 3, 1],
                [2, 3, 1],
            ]
        )

        np.testing.assert_array_almost_equal(grid_rot._freq, freq_expect)
        np.testing.assert_array_almost_equal(grid_rot._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(grid_rot._vals, vals_expect)

    def test_rotate_rad_neg(self):
        freq = np.array([0, 1])
        dirs = np.array([0, 90, 180])
        vals = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=True,
        )

        grid_rot = grid.rotate(-np.pi / 4, degrees=False)

        freq_expect = (2.0 * np.pi) * np.array([0, 1])
        dirs_expect = (np.pi / 180.0) * np.array([45, 135, 225])
        vals_expect = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )

        np.testing.assert_array_almost_equal(grid_rot._freq, freq_expect)
        np.testing.assert_array_almost_equal(grid_rot._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(grid_rot._vals, vals_expect)

    def test__call__(self):
        freq = np.array([0, 1])
        dirs = np.array([0, 90, 180])
        vals = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = grid(freq_hz=True, degrees=True)

        freq_expect = np.array([0, 1])
        dirs_expect = np.array([0, 90, 180])
        vals_expect = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test__call__2(self):
        freq = np.array([0, 1])
        dirs = np.array([0, 90, 180])
        vals = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = grid(freq_hz=False, degrees=False)

        freq_expect = (2.0 * np.pi) * np.array([0, 1])
        dirs_expect = (np.pi / 180.0) * np.array([0, 90, 180])
        vals_expect = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_expect = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        vals_out = grid.interpolate(y, x, freq_hz=True, degrees=True)

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate2(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_expect = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        y_ = (2.0 * np.pi) * y
        x_ = (np.pi / 180.0) * x
        vals_out = grid.interpolate(y_, x_, freq_hz=False, degrees=False)

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_fill_value(self):

        freq = np.array([0, 1, 2])
        dirs = np.array([0, 90, 180, 270])
        vals = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ]
        )
        grid = Grid(freq, dirs, vals, freq_hz=True, degrees=True)

        # extrapolate
        vals_out = grid.interpolate([10, 20], [0, 90], freq_hz=True, degrees=True)

        vals_expect = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_fill_value_None(self):

        freq = np.array([0, 1, 2])
        dirs = np.array([0, 90, 180, 270])
        vals = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ]
        )
        grid = Grid(freq, dirs, vals, freq_hz=True, degrees=True)

        # extrapolate
        vals_out = grid.interpolate(
            [10, 20], [0, 90], freq_hz=True, degrees=True, fill_value=None
        )

        vals_expect = np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
            ]
        )

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_complex_rectangular(self):
        a_real = 7
        b_real = 6
        a_imag = 3
        b_imag = 9

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp_real = np.array([[a_real * x_i + b_real * y_i for x_i in xp] for y_i in yp])
        vp_imag = np.array([[a_imag * x_i + b_imag * y_i for x_i in xp] for y_i in yp])
        vp = vp_real + 1j * vp_imag
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_real_expect = np.array(
            [[a_real * x_i + b_real * y_i for x_i in x] for y_i in y]
        )
        vals_imag_expect = np.array(
            [[a_imag * x_i + b_imag * y_i for x_i in x] for y_i in y]
        )
        vals_expect = vals_real_expect + 1j * vals_imag_expect

        vals_out = grid.interpolate(
            y, x, freq_hz=True, degrees=True, complex_convert="rectangular"
        )

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_complex_polar(self):
        a_amp = 7
        b_amp = 6
        a_phase = 0.01
        b_phase = 0.03

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp_amp = np.array([[a_amp * x_i + b_amp * y_i for x_i in xp] for y_i in yp])
        vp_phase = np.array(
            [[a_phase * x_i + b_phase * y_i for x_i in xp] for y_i in yp]
        )
        vp = vp_amp * (np.cos(vp_phase) + 1j * np.sin(vp_phase))
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_amp_expect = np.array(
            [[a_amp * x_i + b_amp * y_i for x_i in x] for y_i in y]
        )
        vals_phase_expect = np.array(
            [[a_phase * x_i + b_phase * y_i for x_i in x] for y_i in y]
        )
        vals_expect = vals_amp_expect * (
            np.cos(vals_phase_expect) + 1j * np.sin(vals_phase_expect)
        )

        vals_out = grid.interpolate(
            y, x, freq_hz=True, degrees=True, complex_convert="polar"
        )

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_raises_freq(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.zeros((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        with pytest.raises(ValueError):
            grid.interpolate(
                [0, 1, 2, 1], [0, 1, 2]
            )  # freq not monotonically increasing

    def test_interpolate_raises_dirs(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.zeros((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        with pytest.raises(ValueError):
            grid.interpolate(
                [0, 1, 2], [0, 1, 2, 1]
            )  # dirs not monotonically increasing

    def test_interpolate_raises_dirs_outside_bound(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.zeros((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        with pytest.raises(ValueError):
            grid.interpolate(
                [0, 1, 2], [0, 1, 2, 100], degrees=False
            )  # dirs outside bound

    def test_reshape(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        grid_reshaped = grid.reshape(y, x, freq_hz=True, degrees=True)

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * x
        vals_expect = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        freq_out = grid_reshaped._freq
        dirs_out = grid_reshaped._dirs
        vals_out = grid_reshaped._vals

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_reshape2(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        y_ = (2.0 * np.pi) * y
        x_ = (np.pi / 180.0) * x
        grid_reshaped = grid.reshape(y_, x_, freq_hz=False, degrees=False)

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * x
        vals_expect = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        freq_out = grid_reshaped._freq
        dirs_out = grid_reshaped._dirs
        vals_out = grid_reshaped._vals

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_reshape_complex_rectangular(self):
        a_real = 7
        b_real = 6
        a_imag = 3
        b_imag = 9

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp_real = np.array([[a_real * x_i + b_real * y_i for x_i in xp] for y_i in yp])
        vp_imag = np.array([[a_imag * x_i + b_imag * y_i for x_i in xp] for y_i in yp])
        vp = vp_real + 1j * vp_imag
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        grid_reshaped = grid.reshape(
            y, x, freq_hz=True, degrees=True, complex_convert="rectangular"
        )

        freq_out = grid_reshaped._freq
        dirs_out = grid_reshaped._dirs
        vals_out = grid_reshaped._vals

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * x
        vals_real_expect = np.array(
            [[a_real * x_i + b_real * y_i for x_i in x] for y_i in y]
        )
        vals_imag_expect = np.array(
            [[a_imag * x_i + b_imag * y_i for x_i in x] for y_i in y]
        )
        vals_expect = vals_real_expect + 1j * vals_imag_expect

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_reshape_complex_polar(self):
        a_amp = 7
        b_amp = 6
        a_phase = 0.01
        b_phase = 0.03

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp_amp = np.array([[a_amp * x_i + b_amp * y_i for x_i in xp] for y_i in yp])
        vp_phase = np.array(
            [[a_phase * x_i + b_phase * y_i for x_i in xp] for y_i in yp]
        )
        vp = vp_amp * (np.cos(vp_phase) + 1j * np.sin(vp_phase))
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        grid_reshaped = grid.reshape(
            y, x, freq_hz=True, degrees=True, complex_convert="polar"
        )

        freq_out = grid_reshaped._freq
        dirs_out = grid_reshaped._dirs
        vals_out = grid_reshaped._vals

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * x
        vals_amp_expect = np.array(
            [[a_amp * x_i + b_amp * y_i for x_i in x] for y_i in y]
        )
        vals_phase_expect = np.array(
            [[a_phase * x_i + b_phase * y_i for x_i in x] for y_i in y]
        )
        vals_expect = vals_amp_expect * (
            np.cos(vals_phase_expect) + 1j * np.sin(vals_phase_expect)
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_reshape_raises_freq(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.zeros((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        with pytest.raises(ValueError):
            grid.reshape([0, 1, 2, 1], [0, 1, 2])  # freq not monotonically increasing

    def test_reshape_raises_dirs(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.zeros((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        with pytest.raises(ValueError):
            grid.reshape([0, 1, 2], [0, 1, 2, 1])  # dirs not monotonically increasing

    def test__mul__(self):
        freq_in = np.array([1, 2, 3])
        dirs_in = np.array([0, 10, 20, 30])
        vals_in = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ]
        )
        grid = Grid(freq_in, dirs_in, vals_in, degrees=True)

        grid_squared = grid * grid

        vals_expect = np.array(
            [
                [1, 4, 9, 16],
                [1, 4, 9, 16],
                [1, 4, 9, 16],
            ]
        )

        assert isinstance(grid_squared, Grid)
        assert grid_squared._clockwise == grid._clockwise
        assert grid_squared._waves_coming_from == grid._waves_coming_from
        np.testing.assert_array_almost_equal(grid_squared._freq, grid._freq)
        np.testing.assert_array_almost_equal(grid_squared._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(grid_squared._vals, vals_expect)

    def test__mul__raises_type(self, grid):
        with pytest.raises(ValueError):
            grid * grid._vals

    def test__mul__raises_shape(self):
        freq_in = np.array([1, 2, 3])
        dirs_in = np.array([0, 10, 20, 30])
        vals_in = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ]
        )
        grid = Grid(freq_in, dirs_in, vals_in, degrees=True)

        freq_in2 = np.array([1, 2])
        dirs_in2 = np.array([0, 10, 20])
        vals_in2 = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        )
        grid2 = Grid(freq_in2, dirs_in2, vals_in2, degrees=True)

        with pytest.raises(ValueError):
            grid * grid2

    def test__mul__raises_freq(self, grid):
        grid2 = grid.copy()
        grid2._freq = 2.0 * grid2._freq

        with pytest.raises(ValueError):
            grid * grid2

    def test__mul__raises_dirs(self, grid):
        grid2 = grid.copy()
        grid2._dirs = 2.0 * grid2._dirs

        with pytest.raises(ValueError):
            grid * grid2

    def test__mul__raises_convention(self, grid):
        grid.set_wave_convention(clockwise=False, waves_coming_from=False)

        grid2 = grid.copy().set_wave_convention(clockwise=True, waves_coming_from=False)
        with pytest.raises(ValueError):
            grid * grid2

        grid3 = grid.copy().set_wave_convention(
            clockwise=False, waves_coming_from=False
        )
        with pytest.raises(ValueError):
            grid * grid3

    def test__abs__(self):
        freq_in = np.array([1, 2, 3])
        dirs_in = np.array([0, 10, 20])
        vals_in = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [2.0 + 0.0j, 0.0 + 2.0j, -2.0 + 0.0j],
                [3.0 + 0.0j, 0.0 + 3.0j, -3.0 + 0.0j],
            ]
        )
        grid = Grid(
            freq_in,
            dirs_in,
            vals_in,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        grid_abs = abs(grid)

        vals_expect = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )

        np.testing.assert_array_almost_equal(grid_abs._vals, vals_expect)

    def test__repr__(self, grid):
        assert str(grid) == "Grid"


class Test_RAO:
    def test__init__(self):
        freq_in = np.array([0, 1, 2])
        dirs_in = np.array([0, 45, 90, 135])
        vals_in = np.array(
            [
                [(1 + 2j), (3 + 4j), (5 + 6j), (7 + 8j)],
                [(1 + 2j), (3 + 4j), (5 + 6j), (7 + 8j)],
                [(1 + 2j), (3 + 4j), (5 + 6j), (7 + 8j)],
            ]
        )
        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        assert isinstance(rao, RAO)
        assert isinstance(rao, Grid)
        np.testing.assert_array_almost_equal(rao._freq, 2.0 * np.pi * freq_in)
        np.testing.assert_array_almost_equal(rao._dirs, (np.pi / 180.0) * dirs_in)
        np.testing.assert_array_almost_equal(rao._vals, vals_in)
        assert rao._clockwise is True
        assert rao._waves_coming_from is True
        assert rao._freq_hz is True
        assert rao._degrees is True
        assert rao._phase_degrees is False

    def test_from_amp_phase_rad(self):
        freq_in = np.array([0, 1, 2])
        dirs_in = np.array([0, 45, 90, 135])
        amp_in = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ]
        )
        phase_in = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        rao = RAO.from_amp_phase(
            freq_in,
            dirs_in,
            amp_in,
            phase_in,
            phase_degrees=False,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        vals_expect = amp_in * (np.cos(phase_in) + 1j * np.sin(phase_in))

        assert isinstance(rao, RAO)
        assert isinstance(rao, Grid)
        np.testing.assert_array_almost_equal(rao._freq, 2.0 * np.pi * freq_in)
        np.testing.assert_array_almost_equal(rao._dirs, (np.pi / 180.0) * dirs_in)
        np.testing.assert_array_almost_equal(rao._vals, vals_expect)
        assert rao._clockwise is True
        assert rao._waves_coming_from is True
        assert rao._freq_hz is True
        assert rao._degrees is True
        assert rao._phase_degrees is False

    def test_from_amp_phase_deg(self):
        freq_in = np.array([0, 1, 2])
        dirs_in = np.array([0, 45, 90, 135])
        amp_in = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ]
        )
        phase_in = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        rao = RAO.from_amp_phase(
            freq_in,
            dirs_in,
            amp_in,
            phase_in,
            phase_degrees=True,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        vals_expect = amp_in * (
            np.cos((np.pi / 180) * phase_in) + 1j * np.sin((np.pi / 180) * phase_in)
        )

        assert isinstance(rao, RAO)
        assert isinstance(rao, Grid)
        np.testing.assert_array_almost_equal(rao._freq, 2.0 * np.pi * freq_in)
        np.testing.assert_array_almost_equal(rao._dirs, (np.pi / 180.0) * dirs_in)
        np.testing.assert_array_almost_equal(rao._vals, vals_expect)
        assert rao._clockwise is True
        assert rao._waves_coming_from is True
        assert rao._freq_hz is True
        assert rao._degrees is True
        assert rao._phase_degrees is True

    def test_conjugate(self, rao):
        rao_conj = rao.conjugate()

        assert isinstance(rao_conj, RAO)
        np.testing.assert_array_almost_equal(rao_conj._freq, rao._freq)
        np.testing.assert_array_almost_equal(rao_conj._dirs, rao._dirs)
        np.testing.assert_array_almost_equal(rao_conj._vals, rao._vals.conjugate())

    def test_to_amp_phase(self):
        freq_in = np.array([0, 1, 2])
        dirs_in = np.array([0, 1, 2])
        vals_in = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
            ]
        )

        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, amp_out, phase_out = rao.to_amp_phase(
            freq_hz=True, degrees=True, phase_degrees=True
        )

        freq_expect = np.array([0, 1, 2])
        dirs_expect = np.array([0, 1, 2])
        amp_expect = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        phase_expect = np.array(
            [
                [0.0, 90.0, 180.0],
                [0.0, 90.0, 180.0],
                [0.0, 90.0, 180.0],
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(amp_out, amp_expect)
        np.testing.assert_array_almost_equal(phase_out, phase_expect)

    def test_to_amp_phase2(self):
        freq_in = np.array([0, 1, 2])
        dirs_in = np.array([0, 1, 2])
        vals_in = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
            ]
        )

        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, amp_out, phase_out = rao.to_amp_phase(
            freq_hz=False, degrees=False, phase_degrees=False
        )

        freq_expect = (2.0 * np.pi) * np.array([0, 1, 2])
        dirs_expect = (np.pi / 180.0) * np.array([0, 1, 2])
        amp_expect = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        phase_expect = np.array(
            [
                [0.0, np.pi / 2, np.pi],
                [0.0, np.pi / 2, np.pi],
                [0.0, np.pi / 2, np.pi],
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(amp_out, amp_expect)
        np.testing.assert_array_almost_equal(phase_out, phase_expect)

    def test_to_amp_phase_None(self):
        freq_in = np.array([0, 1, 2])
        dirs_in = np.array([0, 1, 2])
        vals_in = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
            ]
        )

        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, amp_out, phase_out = rao.to_amp_phase(
            freq_hz=None, degrees=None, phase_degrees=None
        )

        freq_expect = np.array([0, 1, 2])
        dirs_expect = np.array([0, 1, 2])
        amp_expect = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        phase_expect = np.array(
            [
                [0.0, np.pi / 2.0, np.pi],
                [0.0, np.pi / 2.0, np.pi],
                [0.0, np.pi / 2.0, np.pi],
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(amp_out, amp_expect)
        np.testing.assert_array_almost_equal(phase_out, phase_expect)

    def test__abs__(self, rao):
        out = np.abs(rao)

        assert isinstance(out, Grid)
        assert not isinstance(out, RAO)
        np.testing.assert_array_almost_equal(out._freq, rao._freq)
        np.testing.assert_array_almost_equal(out._dirs, rao._dirs)
        np.testing.assert_array_almost_equal(out._vals, np.abs(rao._vals))
        assert out._clockwise == rao._clockwise
        assert rao._waves_coming_from == rao._waves_coming_from

    def test__repr__(self, rao):
        assert str(rao) == "RAO"


class Test_DirectionalSpectrum:
    def test__init___hz_deg(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        assert isinstance(spectrum, Grid)
        assert spectrum._clockwise is True
        assert spectrum._waves_coming_from is True
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq_in)
        np.testing.assert_array_almost_equal(spectrum._dirs, (np.pi / 180.0) * dirs_in)
        np.testing.assert_array_almost_equal(
            spectrum._vals, 1.0 / (2.0 * np.pi * (np.pi / 180.0)) * vals_in
        )

    def test__init___hz_rad(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = (np.pi / 180.0) * np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=False,
            clockwise=False,
            waves_coming_from=True,
        )

        assert isinstance(spectrum, Grid)
        assert spectrum._clockwise is False
        assert spectrum._waves_coming_from is True
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq_in)
        np.testing.assert_array_almost_equal(spectrum._dirs, dirs_in)
        np.testing.assert_array_almost_equal(
            spectrum._vals, 1.0 / (2.0 * np.pi) * vals_in
        )

    def test__init___rads_deg(self):
        freq_in = (2.0 * np.pi) * np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=False,
            degrees=True,
            clockwise=True,
            waves_coming_from=False,
        )

        assert isinstance(spectrum, Grid)
        assert spectrum._clockwise is True
        assert spectrum._waves_coming_from is False
        np.testing.assert_array_almost_equal(spectrum._freq, freq_in)
        np.testing.assert_array_almost_equal(spectrum._dirs, (np.pi / 180.0) * dirs_in)
        np.testing.assert_array_almost_equal(
            spectrum._vals, 1.0 / (np.pi / 180.0) * vals_in
        )

    def test__init___rads_rad(self):
        freq_in = (2.0 * np.pi) * np.arange(0.0, 1, 0.1)
        dirs_in = (np.pi / 180.0) * np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=False,
            degrees=False,
            clockwise=False,
            waves_coming_from=False,
        )

        assert isinstance(spectrum, Grid)
        assert spectrum._clockwise is False
        assert spectrum._waves_coming_from is False
        np.testing.assert_array_almost_equal(spectrum._freq, freq_in)
        np.testing.assert_array_almost_equal(spectrum._dirs, dirs_in)
        np.testing.assert_array_almost_equal(spectrum._vals, vals_in)

    def test__init__raises_vals_shape(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        values = np.random.random(size=(len(freq), len(dirs) + 1))

        with pytest.raises(ValueError):
            DirectionalSpectrum(freq, dirs, values, freq_hz=True, degrees=True)

    def test__init__raises_vals_neg(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals[0, 1] *= -1

        with pytest.raises(ValueError):
            DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__raises_freq_neg(self):
        freq = np.arange(-0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        values = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalSpectrum(freq, dirs, values, freq_hz=True, degrees=True)

    def test__init__raises_freq_nosort(self):
        freq = np.array([0.5, 0.0, 1.0])
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__raises_dirs_360(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.linspace(0.0, 360.0, 10, endpoint=True)
        vals = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__raises_dirs_2pi(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=True)
        vals = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=False)

    def test__init__raises_dirs_neg(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.linspace(-1.0, 360.0, 10)
        vals = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__repr___(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        assert str(spectrum) == "DirectionalSpectrum"

    def test__call___rads_rad(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum(freq_hz=False, degrees=False)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = (np.pi / 180.0) * dirs_in
        vals_expect = 1.0 / (2.0 * np.pi * (np.pi / 180.0)) * vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test__call___hz_rad(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum(freq_hz=True, degrees=False)

        freq_expect = freq_in
        dirs_expect = (np.pi / 180.0) * dirs_in
        vals_expect = 1.0 / (np.pi / 180.0) * vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test__call___rads_deg(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum(freq_hz=False, degrees=True)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = dirs_in
        vals_expect = 1.0 / (2.0 * np.pi) * vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test__call___hz_deg(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum(freq_hz=True, degrees=True)

        freq_expect = freq_in
        dirs_expect = dirs_in
        vals_expect = vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_hz_deg(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_expect = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        vals_out = spectrum.interpolate(y, x, freq_hz=True, degrees=True)

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_hz_rad(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_expect = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        x *= np.pi / 180.0
        vals_expect /= np.pi / 180.0

        vals_out = spectrum.interpolate(y, x, freq_hz=True, degrees=False)

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_rads_rad(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_expect = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        x *= np.pi / 180.0
        y *= 2.0 * np.pi
        vals_expect /= 2.0 * np.pi * np.pi / 180.0

        vals_out = spectrum.interpolate(y, x, freq_hz=False, degrees=False)

        np.testing.assert_array_almost_equal(vals_out, vals_expect)
