import numpy as np
import pytest

from scarlet_lithium import Grid


@pytest.fixture
def grid():
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
    return grid


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

    def test_wave_convention(self):
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

        convention_expect = {"clockwise": False, "waves_coming_from": True}
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
        vals = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        grid = Grid(freq, dirs, vals, freq_hz=True, degrees=True)

        # extrapolate
        vals_out = grid.interpolate([10, 20], [0, 90], freq_hz=True, degrees=True)

        vals_expect = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_fill_value_None(self):

        freq = np.array([0, 1, 2])
        dirs = np.array([0, 90, 180, 270])
        vals = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        grid = Grid(freq, dirs, vals, freq_hz=True, degrees=True)

        # extrapolate
        vals_out = grid.interpolate([10, 20], [0, 90], freq_hz=True, degrees=True, fill_value=None)

        vals_expect = np.array([
            [1.0, 2.0],
            [1.0, 2.0],
        ])

        np.testing.assert_array_almost_equal(vals_out, vals_expect)
