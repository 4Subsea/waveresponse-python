from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator as RGI

import waveresponse as wr
from waveresponse import (
    RAO,
    CosineFullSpreading,
    CosineHalfSpreading,
    DirectionalBinSpectrum,
    DirectionalSpectrum,
    Grid,
    WaveBinSpectrum,
    WaveSpectrum,
    calculate_response,
    mirror,
)
from waveresponse._core import _check_foldable, _check_is_similar, _GridInterpolator

TEST_PATH = Path(__file__).parent


@pytest.fixture
def freq_dirs():
    """
    Ensure that all grid-like fixtures has these frequency/direction coordinates.
    This is needed for tests to work.
    """
    freq = np.linspace(0, 1.0, 10)
    dirs = np.linspace(0, 360.0, 15, endpoint=False)
    return freq, dirs


@pytest.fixture
def grid(freq_dirs):
    freq, dirs = freq_dirs
    vals = np.random.random((len(freq), len(dirs)))
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
def rao(freq_dirs):
    freq, dirs = freq_dirs
    vals_amp = np.random.random((len(freq), len(dirs)))
    vals_phase = np.random.random((len(freq), len(dirs)))
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


@pytest.fixture
def rao_for_mirroring():
    freq = np.linspace(0, 1.0, 3)
    dirs = np.linspace(0, 180, 3, endpoint=True)
    vals = np.array(
        [
            [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j],
            [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j],
            [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j],
        ]
    )

    return wr.RAO(freq, dirs, vals, degrees=True)


@pytest.fixture
def directional_spectrum(freq_dirs):
    freq, dirs = freq_dirs
    vals = np.random.random((len(freq), len(dirs)))
    spectrum = DirectionalSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=True,
        waves_coming_from=True,
    )
    return spectrum


@pytest.fixture
def directional_bin_spectrum(freq_dirs):
    freq, dirs = freq_dirs
    vals = np.random.random((len(freq), len(dirs)))
    spectrum = DirectionalBinSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=True,
        waves_coming_from=True,
    )
    return spectrum


@pytest.fixture
def wave(freq_dirs):
    freq, dirs = freq_dirs
    vals = np.random.random((len(freq), len(dirs)))
    wave = WaveSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=True,
        waves_coming_from=True,
    )

    return wave


@pytest.fixture
def wavebin(freq_dirs):
    freq, dirs = freq_dirs
    vals = np.random.random((len(freq), len(dirs)))
    wave = WaveBinSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=True,
        waves_coming_from=True,
    )

    return wave


def test_sort():
    dirs_unsorted = [1, 3, 2, 4]
    vals_unsorted = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    dirs_sorted_out, vals_sorted_out = wr._core._sort(dirs_unsorted, vals_unsorted)

    dirs_sorted_expect = np.array([1, 2, 3, 4])
    vals_sorted_expect = np.array(
        [
            [1, 3, 2, 4],
            [1, 3, 2, 4],
        ]
    )

    np.testing.assert_array_almost_equal(dirs_sorted_out, dirs_sorted_expect)
    np.testing.assert_array_almost_equal(vals_sorted_out, vals_sorted_expect)


class Test_multiply:
    def test_rao_and_rao_to_default_grid(self, rao):
        out = wr.multiply(rao, rao.copy())

        vals_expect = rao._vals * rao._vals

        assert isinstance(out, Grid)
        assert not isinstance(out, RAO)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == rao._clockwise
        assert out._waves_coming_from == rao._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, rao._freq)
        np.testing.assert_array_almost_equal(out._dirs, rao._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize("output_type", ("grid", "Grid", Grid))
    def test_grid_and_grid_to_grid(self, output_type, grid):
        out = wr.multiply(grid, grid.copy(), output_type=output_type)

        vals_expect = grid._vals * grid._vals

        assert isinstance(out, Grid)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == grid._clockwise
        assert out._waves_coming_from == grid._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, grid._freq)
        np.testing.assert_array_almost_equal(out._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize("output_type", ("grid", "Grid", Grid))
    def test_rao_and_wave_to_grid(self, output_type, rao, wave):
        out = wr.multiply(rao, wave, output_type=output_type)

        vals_expect = rao._vals * wave._vals

        assert isinstance(out, Grid)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == rao._clockwise
        assert out._waves_coming_from == rao._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, rao._freq)
        np.testing.assert_array_almost_equal(out._dirs, rao._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize("output_type", ("rao", "RAO", RAO))
    def test_rao_and_rao_to_rao(self, output_type, rao):
        out = wr.multiply(rao, rao.copy(), output_type=output_type)

        vals_expect = rao._vals * rao._vals

        assert isinstance(out, RAO)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == rao._clockwise
        assert out._waves_coming_from == rao._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, rao._freq)
        np.testing.assert_array_almost_equal(out._dirs, rao._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize("output_type", ("grid", "Grid", Grid))
    def test_rao_and_rao_to_grid(self, output_type, rao):
        out = wr.multiply(rao, rao.copy(), output_type=output_type)

        vals_expect = rao._vals * rao._vals

        assert isinstance(out, Grid)
        assert not isinstance(out, RAO)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == rao._clockwise
        assert out._waves_coming_from == rao._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, rao._freq)
        np.testing.assert_array_almost_equal(out._dirs, rao._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize(
        "output_type", ("wave_spectrum", "WaveSpectrum", WaveSpectrum)
    )
    def test_wave_and_wave_to_wave(self, output_type, wave):
        out = wr.multiply(wave, wave.copy(), output_type=output_type)

        vals_expect = wave._vals * wave._vals

        assert isinstance(out, WaveSpectrum)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == wave._clockwise
        assert out._waves_coming_from == wave._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, wave._freq)
        np.testing.assert_array_almost_equal(out._dirs, wave._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize(
        "output_type",
        ("directional_spectrum", "DirectionalSpectrum", DirectionalSpectrum),
    )
    def test_wave_and_wave_to_dir_spectrum(self, output_type, wave):
        out = wr.multiply(wave, wave.copy(), output_type=output_type)

        vals_expect = wave._vals * wave._vals

        assert isinstance(out, DirectionalSpectrum)
        assert not isinstance(out, WaveSpectrum)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == wave._clockwise
        assert out._waves_coming_from == wave._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, wave._freq)
        np.testing.assert_array_almost_equal(out._dirs, wave._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize("output_type", ("grid", "Grid", Grid))
    def test_wave_and_wave_to_grid(self, output_type, wave):
        out = wr.multiply(wave, wave.copy(), output_type=output_type)

        vals_expect = wave._vals * wave._vals

        assert isinstance(out, Grid)
        assert not isinstance(out, WaveSpectrum)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == wave._clockwise
        assert out._waves_coming_from == wave._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, wave._freq)
        np.testing.assert_array_almost_equal(out._dirs, wave._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize("output_type", ("grid", "Grid", Grid))
    def test_wavebin_and_wavebin_to_grid(self, output_type, wavebin):
        grid1 = wavebin.copy()
        grid2 = wavebin.copy()
        out = wr.multiply(grid1, grid2, output_type=output_type)

        vals_expect = wavebin._vals * wavebin._vals

        assert isinstance(out, Grid)
        assert not isinstance(out, WaveSpectrum)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == grid1._clockwise
        assert out._waves_coming_from == grid1._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, grid1._freq)
        np.testing.assert_array_almost_equal(out._dirs, grid1._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize("output_type", ("WaveBinSpectrum", WaveBinSpectrum))
    def test_wavebin_and_wavebin_to_wavebin(self, output_type, wavebin):
        grid1 = wavebin.copy()
        grid2 = wavebin.copy()
        out = wr.multiply(grid1, grid2, output_type=output_type)

        vals_expect = wavebin._vals * wavebin._vals

        assert isinstance(out, Grid)
        assert not isinstance(out, WaveSpectrum)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == grid1._clockwise
        assert out._waves_coming_from == grid1._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, grid1._freq)
        np.testing.assert_array_almost_equal(out._dirs, grid1._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize("output_type", ("grid", "Grid", Grid))
    def test_binspectrum_and_binspectrum_to_grid(
        self, output_type, directional_bin_spectrum
    ):
        grid1 = directional_bin_spectrum.copy()
        grid2 = directional_bin_spectrum.copy()
        out = wr.multiply(grid1, grid2, output_type=output_type)

        vals_expect = grid1._vals * grid2._vals

        assert isinstance(out, Grid)
        assert not isinstance(out, WaveSpectrum)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == grid1._clockwise
        assert out._waves_coming_from == grid1._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, grid1._freq)
        np.testing.assert_array_almost_equal(out._dirs, grid1._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    @pytest.mark.parametrize(
        "output_type",
        ("DirectionalBinSpectrum", DirectionalBinSpectrum),
    )
    def test_binspectrum_and_binspectrum_to_binspectrum(
        self, output_type, directional_bin_spectrum
    ):
        grid1 = directional_bin_spectrum.copy()
        grid2 = directional_bin_spectrum.copy()
        out = wr.multiply(grid1, grid2, output_type=output_type)

        vals_expect = grid1._vals * grid2._vals

        assert isinstance(out, Grid)
        assert not isinstance(out, WaveSpectrum)
        assert out._freq_hz is False
        assert out._degrees is False
        assert out._clockwise == grid1._clockwise
        assert out._waves_coming_from == grid1._waves_coming_from
        np.testing.assert_array_almost_equal(out._freq, grid1._freq)
        np.testing.assert_array_almost_equal(out._dirs, grid1._dirs)
        np.testing.assert_array_almost_equal(out._vals, vals_expect)

    def test_raises_output_type(self, grid):
        with pytest.raises(ValueError):
            wr.multiply(grid, grid.copy(), output_type="invalid-type")

    def test_raises_convention(self, grid):
        grid1 = grid.copy()
        grid1.set_wave_convention(clockwise=True)
        grid2 = grid.copy()
        grid2.set_wave_convention(clockwise=False)

        with pytest.raises(ValueError):
            wr.multiply(grid1, grid2, output_type="grid")

    def test_raises_freq_coordinates(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.random.random((10, 15))
        grid1 = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq = np.linspace(0, 1.0, 7)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.random.random((7, 15))
        grid2 = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        with pytest.raises(ValueError):
            wr.multiply(grid1, grid2)

    def test_raises_dirs_coordinates(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.random.random((10, 15))
        grid1 = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 7, endpoint=False)
        vals = np.random.random((10, 7))
        grid2 = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        with pytest.raises(ValueError):
            wr.multiply(grid1, grid2)


class Test_mirror:
    def test_sway(self, rao_for_mirroring):
        rao_out = mirror(rao_for_mirroring, "sway")
        freq_out, dirs_out, vals_out = rao_out.grid()

        freq_expect = rao_for_mirroring.freq()
        dirs_expect = np.linspace(0, 360, 4, endpoint=False)
        vals_expect = np.array(
            [
                [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j, -1.0 + 4.0j],
                [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j, -2.0 - 2.0j],
                [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j, -1.0 + 4.0j],
            ]
        )

        assert isinstance(rao_out, RAO)
        assert rao_out._freq_hz == rao_for_mirroring._freq_hz
        assert rao_out._clockwise == rao_for_mirroring._clockwise
        assert rao_out._waves_coming_from == rao_for_mirroring._waves_coming_from
        assert rao_out._degrees == rao_for_mirroring._degrees
        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_surge(self, rao_for_mirroring):
        rao_out = mirror(rao_for_mirroring, "surge")
        freq_out, dirs_out, vals_out = rao_out.grid()

        freq_expect = rao_for_mirroring.freq()
        dirs_expect = np.linspace(0, 360, 4, endpoint=False)
        vals_expect = np.array(
            [
                [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j, 1.0 - 4.0j],
                [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j, 2.0 + 2.0j],
                [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j, 1.0 - 4.0j],
            ]
        )

        assert isinstance(rao_out, RAO)
        assert rao_out._freq_hz == rao_for_mirroring._freq_hz
        assert rao_out._clockwise == rao_for_mirroring._clockwise
        assert rao_out._waves_coming_from == rao_for_mirroring._waves_coming_from
        assert rao_out._degrees == rao_for_mirroring._degrees
        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_heave(self, rao_for_mirroring):
        rao_out = mirror(rao_for_mirroring, "heave")
        freq_out, dirs_out, vals_out = rao_out.grid()

        freq_expect = rao_for_mirroring.freq()
        dirs_expect = np.linspace(0, 360, 4, endpoint=False)
        vals_expect = np.array(
            [
                [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j, 1.0 - 4.0j],
                [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j, 2.0 + 2.0j],
                [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j, 1.0 - 4.0j],
            ]
        )

        assert isinstance(rao_out, RAO)
        assert rao_out._freq_hz == rao_for_mirroring._freq_hz
        assert rao_out._clockwise == rao_for_mirroring._clockwise
        assert rao_out._waves_coming_from == rao_for_mirroring._waves_coming_from
        assert rao_out._degrees == rao_for_mirroring._degrees
        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_pitch(self, rao_for_mirroring):
        rao_out = mirror(rao_for_mirroring, "pitch")
        freq_out, dirs_out, vals_out = rao_out.grid()

        freq_expect = rao_for_mirroring.freq()
        dirs_expect = np.linspace(0, 360, 4, endpoint=False)
        vals_expect = np.array(
            [
                [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j, 1.0 - 4.0j],
                [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j, 2.0 + 2.0j],
                [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j, 1.0 - 4.0j],
            ]
        )

        assert isinstance(rao_out, RAO)
        assert rao_out._freq_hz == rao_for_mirroring._freq_hz
        assert rao_out._clockwise == rao_for_mirroring._clockwise
        assert rao_out._waves_coming_from == rao_for_mirroring._waves_coming_from
        assert rao_out._degrees == rao_for_mirroring._degrees
        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_roll(self, rao_for_mirroring):
        rao_out = mirror(rao_for_mirroring, "roll")
        freq_out, dirs_out, vals_out = rao_out.grid()

        freq_expect = rao_for_mirroring.freq()
        dirs_expect = np.linspace(0, 360, 4, endpoint=False)
        vals_expect = np.array(
            [
                [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j, -1.0 + 4.0j],
                [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j, -2.0 - 2.0j],
                [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j, -1.0 + 4.0j],
            ]
        )

        assert isinstance(rao_out, RAO)
        assert rao_out._freq_hz == rao_for_mirroring._freq_hz
        assert rao_out._clockwise == rao_for_mirroring._clockwise
        assert rao_out._waves_coming_from == rao_for_mirroring._waves_coming_from
        assert rao_out._degrees == rao_for_mirroring._degrees
        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_yaw(self, rao_for_mirroring):
        rao_out = mirror(rao_for_mirroring, "yaw")
        freq_out, dirs_out, vals_out = rao_out.grid()

        freq_expect = rao_for_mirroring.freq()
        dirs_expect = np.linspace(0, 360, 4, endpoint=False)
        vals_expect = np.array(
            [
                [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j, -1.0 + 4.0j],
                [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j, -2.0 - 2.0j],
                [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j, -1.0 + 4.0j],
            ]
        )

        assert isinstance(rao_out, RAO)
        assert rao_out._freq_hz == rao_for_mirroring._freq_hz
        assert rao_out._clockwise == rao_for_mirroring._clockwise
        assert rao_out._waves_coming_from == rao_for_mirroring._waves_coming_from
        assert rao_out._degrees == rao_for_mirroring._degrees
        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_yaw_rad(self):
        freq = np.linspace(0, 1.0, 3)
        dirs = np.linspace(0, np.pi, 3, endpoint=True)
        vals = np.array(
            [
                [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j],
                [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j],
                [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j],
            ]
        )

        rao_for_mirroring_rad = wr.RAO(freq, dirs, vals, degrees=False)

        rao_out = mirror(rao_for_mirroring_rad, "yaw")
        freq_out, dirs_out, vals_out = rao_out.grid()

        freq_expect = rao_for_mirroring_rad.freq()
        dirs_expect = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        vals_expect = np.array(
            [
                [5.0 + 9.0j, 1.0 - 4.0j, 3.0 - 2.0j, -1.0 + 4.0j],
                [7.0 - 9.0j, 2.0 + 2.0j, 10.0 + 9.0j, -2.0 - 2.0j],
                [6.0 + 1.0j, 1.0 - 4.0j, 7.0 + 3.0j, -1.0 + 4.0j],
            ]
        )

        assert isinstance(rao_out, RAO)
        assert rao_out._freq_hz == rao_for_mirroring_rad._freq_hz
        assert rao_out._clockwise == rao_for_mirroring_rad._clockwise
        assert rao_out._waves_coming_from == rao_for_mirroring_rad._waves_coming_from
        assert rao_out._degrees == rao_for_mirroring_rad._degrees
        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_raises_dirs(self, rao):
        # Try to mirror an RAO with directions greater than 180 degrees
        with pytest.raises(ValueError):
            mirror(rao, "sway", sym_plane="xz")

        with pytest.raises(ValueError):
            mirror(rao, "sway", sym_plane="yz")

    def test_raises_dof(self, rao_for_mirroring):
        with pytest.raises(ValueError):
            mirror(rao_for_mirroring, "invalid-dof")

    mask_bounds = [(0.0, 90.0), (90.0, 180.0), (180.0, 270.0), (270.0, 360.0)]
    sym_plane_order = [("xz", "yz"), ("yz", "xz")]
    dof = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
    params_mirror_twise = product(mask_bounds, sym_plane_order, dof)

    @pytest.mark.parametrize("mask_bounds, sym_plane_order, dof", params_mirror_twise)
    def test_mirror_twise(self, mask_bounds, sym_plane_order, dof):
        """
        Check that we can reconstruct a 'full', symmetric RAO by mirroring the RAO twise
        """
        rao_df = pd.read_csv(
            TEST_PATH / "testdata" / f"rao_{dof}_symmetric.csv", index_col=0
        )
        freq = rao_df.index.astype(float)
        dirs = rao_df.columns.astype(float)
        vals = rao_df.values.astype(complex)
        rao_full = wr.RAO(freq, dirs, vals, freq_hz=False, degrees=False)

        # Construct a 'reduced' version of the RAO, defined only in the range given
        # by the bounds
        freq, dirs, vals = rao_full.grid(freq_hz=False, degrees=True)
        mask = (dirs >= mask_bounds[0]) & (dirs <= mask_bounds[1])
        if mask_bounds[1] == 360.0:
            mask = np.logical_or(mask, dirs == 0.0)
        freq_reduced = freq.copy()
        dirs_reduced = dirs[mask].copy()
        vals_reduced = vals[:, mask].copy()

        rao_reduced = wr.RAO(
            freq_reduced,
            dirs_reduced,
            vals_reduced,
            freq_hz=False,
            degrees=True,
            **rao_full.wave_convention,
        )

        # Mirror the 'reduced' RAO twise to reconstruct the 'full' RAO
        rao_mirrored = wr.mirror(
            wr.mirror(rao_reduced, dof, sym_plane=sym_plane_order[0]),
            dof,
            sym_plane=sym_plane_order[1],
        )

        freq_out, dirs_out, vals_out = rao_mirrored.grid(freq_hz=False, degrees=False)
        freq_expect, dirs_expect, vals_expect = rao_full.grid(
            freq_hz=False, degrees=False
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)


class Test__check_foldable:
    check_foldable_valid = [
        (
            np.linspace(
                np.nextafter(0.0, 180.0),
                np.nextafter(180.0, 0.0),
                num=10,
                endpoint=True,
            ),
            "xz",
        ),
        (
            np.linspace(
                np.nextafter(180.0, 360.0),
                np.nextafter(360.0, 180.0),
                num=10,
                endpoint=True,
            ),
            "xz",
        ),
        (
            np.linspace(
                np.nextafter(90.0, 270.0),
                np.nextafter(270.0, 90.0),
                num=10,
                endpoint=True,
            ),
            "yz",
        ),
        (
            np.linspace(
                np.nextafter(0.0, 90.0), np.nextafter(90.0, 0.0), num=10, endpoint=True
            ),
            "xz",
        ),
        (
            np.linspace(
                np.nextafter(0.0, 90.0), np.nextafter(90.0, 0.0), num=10, endpoint=True
            ),
            "yz",
        ),
        (
            np.linspace(
                np.nextafter(90.0, 180.0),
                np.nextafter(180.0, 90.0),
                num=10,
                endpoint=True,
            ),
            "xz",
        ),
        (
            np.linspace(
                np.nextafter(90.0, 180.0),
                np.nextafter(180.0, 90.0),
                num=10,
                endpoint=True,
            ),
            "yz",
        ),
        (
            np.linspace(
                np.nextafter(180.0, 270.0),
                np.nextafter(270.0, 180.0),
                num=10,
                endpoint=True,
            ),
            "xz",
        ),
        (
            np.linspace(
                np.nextafter(180.0, 270.0),
                np.nextafter(270.0, 180.0),
                num=10,
                endpoint=True,
            ),
            "yz",
        ),
        (
            np.linspace(
                np.nextafter(270.0, 360.0),
                np.nextafter(360.0, 270.0),
                num=10,
                endpoint=True,
            ),
            "yz",
        ),
        (
            np.linspace(
                np.nextafter(270.0, 360.0),
                np.nextafter(360.0, 270.0),
                num=10,
                endpoint=True,
            ),
            "yz",
        ),
    ]

    @pytest.mark.parametrize("dirs, sym_plane", check_foldable_valid)
    def test_is_foldable(self, dirs, sym_plane):
        _check_foldable(dirs, degrees=True, sym_plane=sym_plane)
        _check_foldable(np.radians(dirs), degrees=False, sym_plane=sym_plane)

    check_foldable_invalid = [
        (
            np.linspace(
                np.nextafter(0.0, 180.0),
                np.nextafter(180.0, 360.0),
                num=10,
                endpoint=True,
            ),
            "xz",
        ),
        (
            np.linspace(
                np.nextafter(180.0, 0.0),
                np.nextafter(360.0, 180.0),
                num=10,
                endpoint=True,
            ),
            "xz",
        ),
        (
            np.linspace(
                np.nextafter(90.0, 0.0),
                np.nextafter(270.0, 360.0),
                num=10,
                endpoint=True,
            ),
            "yz",
        ),
    ]

    @pytest.mark.parametrize("dirs, sym_plane", check_foldable_invalid)
    def test_not_foldable(self, dirs, sym_plane):
        with pytest.raises(ValueError):
            _check_foldable(dirs, degrees=True, sym_plane=sym_plane)

        with pytest.raises(ValueError):
            _check_foldable(np.radians(dirs), degrees=False, sym_plane=sym_plane)

    def test_dirs_empty(self):
        with pytest.raises(ValueError):
            _check_foldable(np.array([]), degrees=True, sym_plane="xz")


class Test__GridInterpolator:
    def test_interpolate(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_expect = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        interp_fun = _GridInterpolator(yp * 2.0 * np.pi, np.radians(xp), vp)
        vals_out = interp_fun(y * 2.0 * np.pi, np.radians(x))
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_single_coordinate(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])

        interp_fun = _GridInterpolator(yp * 2.0 * np.pi, np.radians(xp), vp)
        vals_out = interp_fun(1.8 * 2.0 * np.pi, np.radians(12.1))
        vals_expect = np.array(a * 12.1 + b * 1.8)

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
        interp_fun = _GridInterpolator(
            freq * 2.0 * np.pi,
            np.radians(dirs),
            vals,
            fill_value=0.0,
            bounds_error=False,
        )
        vals_out = interp_fun(np.array([10, 20]) * 2.0 * np.pi, np.radians([0, 90]))

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
        interp_fun = _GridInterpolator(
            freq * 2.0 * np.pi,
            np.radians(dirs),
            vals,
            fill_value=None,
            bounds_error=False,
        )
        vals_out = interp_fun(np.array([10, 20]) * 2.0 * np.pi, np.radians([0, 90]))

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

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        vals_real_expect = np.array(
            [[a_real * x_i + b_real * y_i for x_i in x] for y_i in y]
        )
        vals_imag_expect = np.array(
            [[a_imag * x_i + b_imag * y_i for x_i in x] for y_i in y]
        )
        vals_expect = vals_real_expect + 1j * vals_imag_expect

        interp_fun = _GridInterpolator(
            yp * 2.0 * np.pi, np.radians(xp), vp, complex_convert="rectangular"
        )
        vals_out = interp_fun(y * 2.0 * np.pi, np.radians(x))

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

        y = np.linspace(0.0, 2.0, 200)
        x = np.linspace(0.0, 359.0, 100)
        vals_amp_expect = np.array(
            [[a_amp * x_i + b_amp * y_i for x_i in x] for y_i in y]
        )
        x_, y_ = np.meshgrid(x, y, indexing="ij", sparse=True)
        vals_phase_cos_expect = RGI((xp, yp), np.cos(vp_phase).T)((x_, y_)).T
        vals_phase_sin_expect = RGI((xp, yp), np.sin(vp_phase).T)((x_, y_)).T

        vals_expect = (
            vals_amp_expect
            * (vals_phase_cos_expect + 1j * vals_phase_sin_expect)
            / np.abs(vals_phase_cos_expect + 1j * vals_phase_sin_expect)
        )

        interp_fun = _GridInterpolator(
            yp * 2.0 * np.pi, np.radians(xp), vp, complex_convert="polar"
        )
        vals_out = interp_fun(y * 2.0 * np.pi, np.radians(x))

        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate_raises_outside_bound(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.zeros((10, 15))
        interp_fun = _GridInterpolator(
            freq * 2.0 * np.pi,
            np.radians(dirs),
            vals,
            bounds_error=True,
        )

        with pytest.raises(ValueError):
            interp_fun(np.array([0, 0.5]) * 2.0 * np.pi, np.radians([0, 1, 2, 400]))

        with pytest.raises(ValueError):
            interp_fun(np.array([0, 2.0]) * 2.0 * np.pi, np.radians([0, 1, 2, 100]))


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

    def test_interpolate_single_coordinate(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        grid = Grid(yp, xp, vp, freq_hz=True, degrees=True)

        vals_out = grid.interpolate(1.8, 12.1, freq_hz=True, degrees=True)

        vals_expect = np.array(a * 12.1 + b * 1.8)

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

        y = np.linspace(0.0, 2.0, 200)
        x = np.linspace(0.0, 359.0, 100)
        vals_amp_expect = np.array(
            [[a_amp * x_i + b_amp * y_i for x_i in x] for y_i in y]
        )
        x_, y_ = np.meshgrid(x, y, indexing="ij", sparse=True)
        vals_phase_cos_expect = RGI((xp, yp), np.cos(vp_phase).T)((x_, y_)).T
        vals_phase_sin_expect = RGI((xp, yp), np.sin(vp_phase).T)((x_, y_)).T

        vals_expect = (
            vals_amp_expect
            * (vals_phase_cos_expect + 1j * vals_phase_sin_expect)
            / np.abs(vals_phase_cos_expect + 1j * vals_phase_sin_expect)
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
        x_, y_ = np.meshgrid(x, y, indexing="ij", sparse=True)
        vals_phase_cos_expect = RGI((xp, yp), np.cos(vp_phase).T)((x_, y_)).T
        vals_phase_sin_expect = RGI((xp, yp), np.sin(vp_phase).T)((x_, y_)).T

        vals_expect = (
            vals_amp_expect
            * (vals_phase_cos_expect + 1j * vals_phase_sin_expect)
            / np.abs(vals_phase_cos_expect + 1j * vals_phase_sin_expect)
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

    def test__repr__(self, grid):
        assert str(grid) == "Grid"

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

    def test__init__raises_freq_not_1d(self):
        with pytest.raises(ValueError, match="`freq` must be 1 dimensional."):
            freq = np.array([[0, 1], [2, 3]])
            dirs = np.array([0.0, 1.0, 1.5, 2.0])
            vals = np.zeros((4, 4))
            Grid(freq, dirs, vals)

    def test__init__raises_dirs_not_1d(self):
        with pytest.raises(ValueError, match="`dirs` must be 1 dimensional."):
            freq = np.array([0, 1, 2, 3])
            dirs = np.array([[0.0, 1.0], [1.5, 2.0]])
            vals = np.zeros((4, 4))
            Grid(freq, dirs, vals)

    def test__init__raises_vals_shape(self):
        with pytest.raises(ValueError):
            freq = np.array([0, 1, 2])
            dirs = np.array([0, 1, 2, 3])
            vals = np.zeros((3, 10))
            Grid(freq, dirs, vals)

    def test_from_grid(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.random.random((10, 15))
        grid_in = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        grid_out = Grid.from_grid(grid_in)

        vals_expect = vals.copy()

        assert isinstance(grid_out, Grid)
        np.testing.assert_array_almost_equal(grid_out._freq, grid_in._freq)
        np.testing.assert_array_almost_equal(grid_out._dirs, grid_in._dirs)
        np.testing.assert_array_almost_equal(grid_out._vals, vals_expect)
        assert grid_out._clockwise == grid_in._clockwise
        assert grid_out._waves_coming_from == grid_in._waves_coming_from
        assert grid_out._freq_hz == grid_in._freq_hz
        assert grid_out._degrees == grid_in._degrees

    def test_freq_None(self):
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

        freq_out = grid.freq()
        np.testing.assert_array_almost_equal(freq_out, freq)

    def test_freq_rads(self):
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

        freq_out = grid.freq(freq_hz=False)
        np.testing.assert_array_almost_equal(freq_out, (2.0 * np.pi) * freq)

    def test_freq_hz(self):
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

        freq_out = grid.freq(freq_hz=True)
        np.testing.assert_array_almost_equal(freq_out, freq)

    def test_dirs_None(self):
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

        dirs_out = grid.dirs()
        np.testing.assert_array_almost_equal(dirs_out, dirs)

    def test_dirs_rad(self):
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

        dirs_out = grid.dirs(degrees=False)
        np.testing.assert_array_almost_equal(dirs_out, (np.pi / 180.0) * dirs)

    def test_dirs_deg(self):
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

        dirs_out = grid.dirs(degrees=True)
        np.testing.assert_array_almost_equal(dirs_out, dirs)

    def test_wave_convention(self, grid):
        convention_expect = {"clockwise": True, "waves_coming_from": True}
        convention_out = grid.wave_convention
        assert convention_out == convention_expect

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

    def test_grid(self):
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

        freq_out, dirs_out, vals_out = grid.grid(freq_hz=True, degrees=True)

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

    def test__grid2(self):
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

        freq_out, dirs_out, vals_out = grid.grid(freq_hz=False, degrees=False)

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

    def test__mul__numeric(self, grid):
        grid_scaled = grid * 2.0
        np.testing.assert_array_almost_equal(grid_scaled._vals, grid._vals * 2.0)

        grid_scaled = grid * -2.0
        np.testing.assert_array_almost_equal(grid_scaled._vals, grid._vals * -2.0)

        grid_scaled = grid * 0.0
        np.testing.assert_array_almost_equal(grid_scaled._vals, grid._vals * 0.0)

        grid_scaled = grid * 2
        np.testing.assert_array_almost_equal(grid_scaled._vals, grid._vals * 2)

        grid_scaled = grid * (1 + 1j)
        np.testing.assert_array_almost_equal(grid_scaled._vals, grid._vals * (1 + 1j))

    def test__rmul__numeric(self, grid):
        grid_scaled = 2.0 * grid
        np.testing.assert_array_almost_equal(grid_scaled._vals, grid._vals * 2.0)

    def test__mul__raises_array(self, grid):
        with pytest.raises(TypeError):
            grid * grid._vals

    def test__mul__raises_type(self, grid, rao):
        with pytest.raises(TypeError):
            grid * rao

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

        grid2 = grid.copy()
        grid2.set_wave_convention(clockwise=True, waves_coming_from=False)
        with pytest.raises(ValueError):
            grid * grid2

        grid3 = grid.copy()
        grid3.set_wave_convention(clockwise=False, waves_coming_from=True)
        with pytest.raises(ValueError):
            grid * grid3

    def test__add__(self, grid):
        out = grid + grid

        assert isinstance(out, Grid)
        np.testing.assert_array_almost_equal(out._vals, grid._vals + grid._vals)

    def test__add__raises_type(self, grid, rao):
        with pytest.raises(TypeError):
            grid + rao

    @patch("waveresponse._core._check_is_similar")
    def test__add__check_is_similar(self, mock_check_is_similar, grid):
        grid + grid
        mock_check_is_similar.assert_called_once_with(grid, grid, exact_type=True)

    def test__add__numeric(self, grid):
        grid_added = grid + 2.0
        np.testing.assert_array_almost_equal(grid_added._vals, grid._vals + 2.0)

        grid_added = grid + 0.0
        np.testing.assert_array_almost_equal(grid_added._vals, grid._vals + 0.0)

        grid_added = grid + 2
        np.testing.assert_array_almost_equal(grid_added._vals, grid._vals + 2)

        grid_added = grid + (1 + 1j)
        np.testing.assert_array_almost_equal(grid_added._vals, grid._vals + (1 + 1j))

    def test__radd__numeric(self, grid):
        grid_added = 2.0 + grid
        np.testing.assert_array_almost_equal(grid_added._vals, grid._vals + 2.0)

    def test__sub__(self, grid):
        out = grid - grid

        assert isinstance(out, Grid)
        np.testing.assert_array_almost_equal(out._vals, grid._vals - grid._vals)

    @patch("waveresponse._core._check_is_similar")
    def test__sub__check_is_similar(self, mock_check_is_similar, grid):
        grid - grid
        mock_check_is_similar.assert_called_once_with(grid, grid, exact_type=True)

    def test__sub__raises_type(self, grid, rao):
        with pytest.raises(TypeError):
            grid - rao

    def test__sub__numeric(self, grid):
        grid_subtracted = grid - 2.0
        np.testing.assert_array_almost_equal(grid_subtracted._vals, grid._vals - 2.0)

        grid_subtracted = grid - 0.0
        np.testing.assert_array_almost_equal(grid_subtracted._vals, grid._vals - 0.0)

        grid_subtracted = grid - 2
        np.testing.assert_array_almost_equal(grid_subtracted._vals, grid._vals - 2)

        grid_subtracted = grid - (1 + 1j)
        np.testing.assert_array_almost_equal(
            grid_subtracted._vals, grid._vals - (1 + 1j)
        )

    def test__rsub__numeric(self, grid):
        grid_subtracted = 2.0 - grid
        np.testing.assert_array_almost_equal(grid_subtracted._vals, grid._vals - 2.0)

    def test_conjugate(self, grid):
        grid_conj = grid.conjugate()

        np.testing.assert_array_almost_equal(grid_conj._freq, grid._freq)
        np.testing.assert_array_almost_equal(grid_conj._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(grid_conj._vals, grid._vals.conjugate())

    def test_real(self):
        freq_in = np.array([1, 2, 3])
        dirs_in = np.array([0, 10, 20])
        vals_in = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [2.0 + 0.0j, 0.0 - 2.0j, -2.0 + 0.0j],
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

        grid_real = grid.real

        vals_expect = np.array(
            [
                [1.0, 0.0, -1.0],
                [2.0, 0.0, -2.0],
                [3.0, 0.0, -3.0],
            ]
        )

        assert isinstance(grid_real, Grid)
        np.testing.assert_array_almost_equal(grid_real._vals, vals_expect)

    def test_imag(self):
        freq_in = np.array([1, 2, 3])
        dirs_in = np.array([0, 10, 20])
        vals_in = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [2.0 + 0.0j, 0.0 - 2.0j, -2.0 + 0.0j],
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

        grid_imag = grid.imag

        vals_expect = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )

        assert isinstance(grid_imag, Grid)
        np.testing.assert_array_almost_equal(grid_imag._vals, vals_expect)


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
        assert rao._phase_leading is True

    def test_from_grid(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.random.random((10, 15)) + 1j * np.random.random((10, 15))
        grid_in = Grid(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        rao_out = RAO.from_grid(grid_in)

        vals_expect = vals.copy()

        assert isinstance(rao_out, RAO)
        np.testing.assert_array_almost_equal(rao_out._freq, grid_in._freq)
        np.testing.assert_array_almost_equal(rao_out._dirs, grid_in._dirs)
        np.testing.assert_array_almost_equal(rao_out._vals, vals_expect)
        assert rao_out._clockwise == grid_in._clockwise
        assert rao_out._waves_coming_from == grid_in._waves_coming_from
        assert rao_out._freq_hz == grid_in._freq_hz
        assert rao_out._degrees == grid_in._degrees

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
            phase_leading=True,
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
        assert rao._phase_leading is True

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
        assert rao._phase_leading is True

    def test_from_amp_phase_lagging(self):
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
            phase_leading=False,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        vals_expect = amp_in * np.exp(-1j * phase_in)

        np.testing.assert_array_almost_equal(rao._freq, 2.0 * np.pi * freq_in)
        np.testing.assert_array_almost_equal(rao._dirs, (np.pi / 180.0) * dirs_in)
        np.testing.assert_array_almost_equal(rao._vals, vals_expect)
        assert rao._phase_leading is False

    def test__mul__(self, rao):
        rao_squared = rao * rao

        assert isinstance(rao_squared, RAO)
        assert rao_squared._freq_hz == rao._freq_hz
        assert rao_squared._degrees == rao._degrees
        assert rao_squared._clockwise == rao._clockwise
        assert rao_squared._waves_coming_from == rao._waves_coming_from
        np.testing.assert_array_almost_equal(rao_squared._freq, rao._freq)
        np.testing.assert_array_almost_equal(rao_squared._dirs, rao._dirs)
        np.testing.assert_array_almost_equal(rao_squared._vals, rao._vals * rao._vals)

    def test__mul__raises_type(self, rao, wave):
        with pytest.raises(TypeError):
            rao * wave

    def test_conjugate(self, rao):
        rao_conj = rao.conjugate()

        assert isinstance(rao_conj, RAO)
        np.testing.assert_array_almost_equal(rao_conj._freq, rao._freq)
        np.testing.assert_array_almost_equal(rao_conj._dirs, rao._dirs)
        np.testing.assert_array_almost_equal(rao_conj._vals, rao._vals.conjugate())

    def test_differentiate(self):
        freq_in = np.array([0, 1.0, 2.0, 3.0])
        dirs_in = np.array([0, 45, 90, 135])
        vals_in = np.array(
            [
                [(1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j)],
                [(0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j)],
                [(1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j)],
                [(0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j)],
            ]
        )
        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=False,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        rao_diff = rao.differentiate()

        freq_expect = rao._freq
        dirs_expect = rao._dirs
        vals_expect = np.array(
            [
                [
                    0.0j * (1.0 + 0.0j),
                    0.0j * (1.0 + 0.0j),
                    0.0j * (1.0 + 0.0j),
                    0.0j * (1.0 + 0.0j),
                ],
                [
                    1.0j * (0.0 + 1.0j),
                    1.0j * (0.0 + 1.0j),
                    1.0j * (0.0 + 1.0j),
                    1.0j * (0.0 + 1.0j),
                ],
                [
                    2.0j * (1.0 + 0.0j),
                    2.0j * (1.0 + 0.0j),
                    2.0j * (1.0 + 0.0j),
                    2.0j * (1.0 + 0.0j),
                ],
                [
                    3.0j * (0.0 + 1.0j),
                    3.0j * (0.0 + 1.0j),
                    3.0j * (0.0 + 1.0j),
                    3.0j * (0.0 + 1.0j),
                ],
            ]
        )

        assert isinstance(rao_diff, RAO)
        np.testing.assert_array_almost_equal(rao_diff._freq, freq_expect)
        np.testing.assert_array_almost_equal(rao_diff._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(rao_diff._vals, vals_expect)
        assert rao._clockwise == rao._clockwise
        assert rao._waves_coming_from == rao._waves_coming_from

    def test_differentiate_order2(self):
        freq_in = np.array([0, 1.0, 2.0, 3.0])
        dirs_in = np.array([0, 45, 90, 135])
        vals_in = np.array(
            [
                [(1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j)],
                [(0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j)],
                [(1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j)],
                [(0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j)],
            ]
        )
        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=False,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        rao_diff = rao.differentiate(2)

        freq_expect = rao._freq
        dirs_expect = rao._dirs
        vals_expect = np.array(
            [
                [
                    0.0j**2 * (1.0 + 0.0j),
                    0.0j**2 * (1.0 + 0.0j),
                    0.0j**2 * (1.0 + 0.0j),
                    0.0j**2 * (1.0 + 0.0j),
                ],
                [
                    1.0j**2 * (0.0 + 1.0j),
                    1.0j**2 * (0.0 + 1.0j),
                    1.0j**2 * (0.0 + 1.0j),
                    1.0j**2 * (0.0 + 1.0j),
                ],
                [
                    2.0j**2 * (1.0 + 0.0j),
                    2.0j**2 * (1.0 + 0.0j),
                    2.0j**2 * (1.0 + 0.0j),
                    2.0j**2 * (1.0 + 0.0j),
                ],
                [
                    3.0j**2 * (0.0 + 1.0j),
                    3.0j**2 * (0.0 + 1.0j),
                    3.0j**2 * (0.0 + 1.0j),
                    3.0j**2 * (0.0 + 1.0j),
                ],
            ]
        )

        assert isinstance(rao_diff, RAO)
        np.testing.assert_array_almost_equal(rao_diff._freq, freq_expect)
        np.testing.assert_array_almost_equal(rao_diff._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(rao_diff._vals, vals_expect)
        assert rao._clockwise == rao._clockwise
        assert rao._waves_coming_from == rao._waves_coming_from

    def test_differentiate_order3(self):
        freq_in = np.array([0, 1.0, 2.0, 3.0])
        dirs_in = np.array([0, 45, 90, 135])
        vals_in = np.array(
            [
                [(1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j)],
                [(0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j)],
                [(1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j), (1.0 + 0.0j)],
                [(0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j), (0.0 + 1.0j)],
            ]
        )
        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=False,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        rao_diff = rao.differentiate(3)

        freq_expect = rao._freq
        dirs_expect = rao._dirs
        vals_expect = np.array(
            [
                [
                    0.0j**3 * (1.0 + 0.0j),
                    0.0j**3 * (1.0 + 0.0j),
                    0.0j**3 * (1.0 + 0.0j),
                    0.0j**3 * (1.0 + 0.0j),
                ],
                [
                    1.0j**3 * (0.0 + 1.0j),
                    1.0j**3 * (0.0 + 1.0j),
                    1.0j**3 * (0.0 + 1.0j),
                    1.0j**3 * (0.0 + 1.0j),
                ],
                [
                    2.0j**3 * (1.0 + 0.0j),
                    2.0j**3 * (1.0 + 0.0j),
                    2.0j**3 * (1.0 + 0.0j),
                    2.0j**3 * (1.0 + 0.0j),
                ],
                [
                    3.0j**3 * (0.0 + 1.0j),
                    3.0j**3 * (0.0 + 1.0j),
                    3.0j**3 * (0.0 + 1.0j),
                    3.0j**3 * (0.0 + 1.0j),
                ],
            ]
        )

        assert isinstance(rao_diff, RAO)
        np.testing.assert_array_almost_equal(rao_diff._freq, freq_expect)
        np.testing.assert_array_almost_equal(rao_diff._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(rao_diff._vals, vals_expect)
        assert rao._clockwise == rao._clockwise
        assert rao._waves_coming_from == rao._waves_coming_from

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

    def test_to_amp_phase_lagging(self):
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
            freq_hz=True, degrees=True, phase_degrees=True, phase_leading=False
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
                [0.0, -90.0, 180.0],
                [0.0, -90.0, 180.0],
                [0.0, -90.0, 180.0],
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(amp_out, amp_expect)
        np.testing.assert_array_almost_equal(phase_out, phase_expect)

    def test__repr__(self, rao):
        assert str(rao) == "RAO"

    def test_real(self):
        freq_in = np.array([1, 2, 3])
        dirs_in = np.array([0, 10, 20])
        vals_in = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [2.0 + 0.0j, 0.0 - 2.0j, -2.0 + 0.0j],
                [3.0 + 0.0j, 0.0 + 3.0j, -3.0 + 0.0j],
            ]
        )
        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        grid_real = rao.real

        vals_expect = np.array(
            [
                [1.0, 0.0, -1.0],
                [2.0, 0.0, -2.0],
                [3.0, 0.0, -3.0],
            ]
        )

        assert isinstance(grid_real, Grid)
        assert not isinstance(grid_real, RAO)
        np.testing.assert_array_almost_equal(grid_real._vals, vals_expect)

    def test_imag(self):
        freq_in = np.array([1, 2, 3])
        dirs_in = np.array([0, 10, 20])
        vals_in = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
                [2.0 + 0.0j, 0.0 - 2.0j, -2.0 + 0.0j],
                [3.0 + 0.0j, 0.0 + 3.0j, -3.0 + 0.0j],
            ]
        )
        rao = RAO(
            freq_in,
            dirs_in,
            vals_in,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        grid_imag = rao.imag

        vals_expect = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )

        assert isinstance(grid_imag, Grid)
        assert not isinstance(grid_imag, RAO)
        np.testing.assert_array_almost_equal(grid_imag._vals, vals_expect)


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

    def test__init__vals_neg_ok(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals[0, 1] *= -1

        _ = DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__vals_complex_ok(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals = vals + 1j * vals

        _ = DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

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

    def test__repr___(self, directional_spectrum):
        assert str(directional_spectrum) == "DirectionalSpectrum"

    def test_grid_rads_rad(self):
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

        freq_out, dirs_out, vals_out = spectrum.grid(freq_hz=False, degrees=False)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = (np.pi / 180.0) * dirs_in
        vals_expect = 1.0 / (2.0 * np.pi * (np.pi / 180.0)) * vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_grid_hz_rad(self):
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

        freq_out, dirs_out, vals_out = spectrum.grid(freq_hz=True, degrees=False)

        freq_expect = freq_in
        dirs_expect = (np.pi / 180.0) * dirs_in
        vals_expect = 1.0 / (np.pi / 180.0) * vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_grid_rads_deg(self):
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

        freq_out, dirs_out, vals_out = spectrum.grid(freq_hz=False, degrees=True)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = dirs_in
        vals_expect = 1.0 / (2.0 * np.pi) * vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_grid__hz_deg(self):
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

        freq_out, dirs_out, vals_out = spectrum.grid(freq_hz=True, degrees=True)

        freq_expect = freq_in
        dirs_expect = dirs_in
        vals_expect = vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_bingrid(self):
        freq_in = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        dirs_in = np.array([0.0, 90.0, 180.0])
        vals_in = np.ones((len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.bingrid(freq_hz=False, degrees=False)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = (np.pi / 180.0) * dirs_in
        vals_expect = 1.0 / (2.0 * np.pi * (np.pi / 180.0)) * vals_in

        vals_expect[:, 0] *= 135.0 * (np.pi / 180.0)
        vals_expect[:, 1] *= 90.0 * (np.pi / 180.0)
        vals_expect[:, 2] *= 135.0 * (np.pi / 180.0)

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_bingrid2(self):
        freq_in = np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * 2.0 * np.pi
        dirs_in = np.radians(np.array([0.0, 90.0, 180.0]))
        vals_in = np.ones((len(freq_in), len(dirs_in)))
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=False,
            degrees=False,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.bingrid(freq_hz=False, degrees=False)

        freq_expect = freq_in
        dirs_expect = dirs_in
        vals_expect = vals_in

        vals_expect[:, 0] *= 135.0 * (np.pi / 180.0)
        vals_expect[:, 1] *= 90.0 * (np.pi / 180.0)
        vals_expect[:, 2] *= 135.0 * (np.pi / 180.0)

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    @pytest.mark.parametrize(
        "dirs_in",
        [
            np.array([0.0, 90.0, 180.0, 270.0]),
            np.array([10.0, 100.0, 190.0, 280.0]),
            np.array([85.0, 175.0, 265.0, 355.0]),
        ],
    )
    def test_bingrid_rads_rad(self, dirs_in):
        freq_in = np.arange(0.0, 1, 0.1)
        vals_in = np.column_stack(
            [
                np.zeros_like(freq_in),
                np.ones_like(freq_in),
                4 * np.ones_like(freq_in),
                np.ones_like(freq_in),
            ]
        )
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.bingrid(freq_hz=False, degrees=False)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = (np.pi / 180.0) * dirs_in
        vals_expect = np.column_stack(
            [
                22.5 * np.ones_like(freq_in),
                112.5 * np.ones_like(freq_in),
                292.5 * np.ones_like(freq_in),
                112.5 * np.ones_like(freq_in),
            ]
        ) / (2.0 * np.pi)

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    @pytest.mark.parametrize(
        "dirs_in",
        [
            np.array([0.0, 90.0, 180.0, 270.0]),
            np.array([10.0, 100.0, 190.0, 280.0]),
            np.array([85.0, 175.0, 265.0, 355.0]),
        ],
    )
    def test_bingrid_hz_rad(self, dirs_in):
        freq_in = np.arange(0.0, 1, 0.1)
        vals_in = np.column_stack(
            [
                np.zeros_like(freq_in),
                np.ones_like(freq_in),
                4 * np.ones_like(freq_in),
                np.ones_like(freq_in),
            ]
        )

        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.bingrid(freq_hz=True, degrees=False)

        freq_expect = freq_in
        dirs_expect = dirs_in * (np.pi / 180.0)
        vals_expect = np.column_stack(
            [
                22.5 * np.ones_like(freq_in),
                112.5 * np.ones_like(freq_in),
                292.5 * np.ones_like(freq_in),
                112.5 * np.ones_like(freq_in),
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    @pytest.mark.parametrize(
        "dirs_in",
        [
            np.array([0.0, 90.0, 180.0, 270.0]),
            np.array([10.0, 100.0, 190.0, 280.0]),
            np.array([85.0, 175.0, 265.0, 355.0]),
        ],
    )
    def test_bingrid_rads_deg(self, dirs_in):
        freq_in = np.arange(0.0, 1, 0.1)
        vals_in = np.column_stack(
            [
                np.zeros_like(freq_in),
                np.ones_like(freq_in),
                4 * np.ones_like(freq_in),
                np.ones_like(freq_in),
            ]
        )

        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.bingrid(freq_hz=False, degrees=True)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = dirs_in
        vals_expect = np.column_stack(
            [
                22.5 * np.ones_like(freq_in),
                112.5 * np.ones_like(freq_in),
                292.5 * np.ones_like(freq_in),
                112.5 * np.ones_like(freq_in),
            ]
        ) / (2.0 * np.pi)

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    @pytest.mark.parametrize(
        "dirs_in",
        [
            np.array([0.0, 90.0, 180.0, 270.0]),
            np.array([10.0, 100.0, 190.0, 280.0]),
            np.array([85.0, 175.0, 265.0, 355.0]),
        ],
    )
    def test_bingrid_hz_deg(self, dirs_in):
        freq_in = np.arange(0.0, 1, 0.1)
        vals_in = np.column_stack(
            [
                np.zeros_like(freq_in),
                np.ones_like(freq_in),
                4 * np.ones_like(freq_in),
                np.ones_like(freq_in),
            ]
        )
        spectrum = DirectionalSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.bingrid(freq_hz=True, degrees=True)

        freq_expect = freq_in
        dirs_expect = dirs_in
        dirs_expect = dirs_in
        vals_expect = np.column_stack(
            [
                22.5 * np.ones_like(freq_in),
                112.5 * np.ones_like(freq_in),
                292.5 * np.ones_like(freq_in),
                112.5 * np.ones_like(freq_in),
            ]
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_from_spectrum1d(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 90.0, 180.0, 270.0])
        spectrum1d = np.array([1.0, 2.0, 3.0])

        def spread_fun(f, d):
            return np.cos(np.radians(d) / 2.0) ** 2

        spectrum = DirectionalSpectrum.from_spectrum1d(
            freq, dirs, spectrum1d, spread_fun, 45.0, freq_hz=False, degrees=True
        )

        vals_expect = np.array(
            [
                [
                    np.cos(1 * np.pi / 8) ** 2 * 1.0,
                    np.cos(7 * np.pi / 8) ** 2 * 1.0,
                    np.cos(3 * np.pi / 8) ** 2 * 1.0,
                    np.cos(5 * np.pi / 8) ** 2 * 1.0,
                ],
                [
                    np.cos(1 * np.pi / 8) ** 2 * 2.0,
                    np.cos(7 * np.pi / 8) ** 2 * 2.0,
                    np.cos(3 * np.pi / 8) ** 2 * 2.0,
                    np.cos(5 * np.pi / 8) ** 2 * 2.0,
                ],
                [
                    np.cos(1 * np.pi / 8) ** 2 * 3.0,
                    np.cos(7 * np.pi / 8) ** 2 * 3.0,
                    np.cos(3 * np.pi / 8) ** 2 * 3.0,
                    np.cos(5 * np.pi / 8) ** 2 * 3.0,
                ],
            ]
        )
        vals_expect = vals_expect / (np.pi / 180.0)

        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)

    def test_from_spectrum1d_radians(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = (np.pi / 180.0) * np.array([0.0, 90.0, 180.0, 270.0])
        spectrum1d = np.array([1.0, 2.0, 3.0])

        def spread_fun(f, d):
            return np.cos(d / 2.0) ** 2

        spectrum = DirectionalSpectrum.from_spectrum1d(
            freq, dirs, spectrum1d, spread_fun, np.pi / 4, freq_hz=False, degrees=False
        )

        vals_expect = np.array(
            [
                [
                    np.cos(1 * np.pi / 8) ** 2 * 1.0,
                    np.cos(7 * np.pi / 8) ** 2 * 1.0,
                    np.cos(3 * np.pi / 8) ** 2 * 1.0,
                    np.cos(5 * np.pi / 8) ** 2 * 1.0,
                ],
                [
                    np.cos(1 * np.pi / 8) ** 2 * 2.0,
                    np.cos(7 * np.pi / 8) ** 2 * 2.0,
                    np.cos(3 * np.pi / 8) ** 2 * 2.0,
                    np.cos(5 * np.pi / 8) ** 2 * 2.0,
                ],
                [
                    np.cos(1 * np.pi / 8) ** 2 * 3.0,
                    np.cos(7 * np.pi / 8) ** 2 * 3.0,
                    np.cos(3 * np.pi / 8) ** 2 * 3.0,
                    np.cos(5 * np.pi / 8) ** 2 * 3.0,
                ],
            ]
        )

        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)

    def test_from_spectrum1d_neg(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 90.0, 180.0, 270.0])
        spectrum1d = np.array([1.0, 2.0, 3.0])

        def spread_fun(f, d):
            return np.cos(np.radians(d) / 2.0) ** 2

        spectrum = DirectionalSpectrum.from_spectrum1d(
            freq, dirs, spectrum1d, spread_fun, -45.0, freq_hz=False, degrees=True
        )

        vals_expect = np.array(
            [
                [
                    np.cos(1 * np.pi / 8) ** 2 * 1.0,
                    np.cos(3 * np.pi / 8) ** 2 * 1.0,
                    np.cos(5 * np.pi / 8) ** 2 * 1.0,
                    np.cos(7 * np.pi / 8) ** 2 * 1.0,
                ],
                [
                    np.cos(1 * np.pi / 8) ** 2 * 2.0,
                    np.cos(3 * np.pi / 8) ** 2 * 2.0,
                    np.cos(5 * np.pi / 8) ** 2 * 2.0,
                    np.cos(7 * np.pi / 8) ** 2 * 2.0,
                ],
                [
                    np.cos(1 * np.pi / 8) ** 2 * 3.0,
                    np.cos(3 * np.pi / 8) ** 2 * 3.0,
                    np.cos(5 * np.pi / 8) ** 2 * 3.0,
                    np.cos(7 * np.pi / 8) ** 2 * 3.0,
                ],
            ]
        )
        vals_expect = vals_expect / (np.pi / 180.0)

        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)

    def test_from_spectrum1d_spread_freq(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 90.0, 180.0, 270.0])
        spectrum1d = np.array([1.0, 2.0, 3.0])

        def spread_fun(f, d):
            return f * np.cos(np.radians(d) / 2.0) ** 2

        spectrum = DirectionalSpectrum.from_spectrum1d(
            freq, dirs, spectrum1d, spread_fun, 45.0, freq_hz=False, degrees=True
        )

        vals_expect = np.array(
            [
                [
                    0.0 * np.cos(1 * np.pi / 8) ** 2 * 1.0,
                    0.0 * np.cos(7 * np.pi / 8) ** 2 * 1.0,
                    0.0 * np.cos(3 * np.pi / 8) ** 2 * 1.0,
                    0.0 * np.cos(5 * np.pi / 8) ** 2 * 1.0,
                ],
                [
                    0.5 * np.cos(1 * np.pi / 8) ** 2 * 2.0,
                    0.5 * np.cos(7 * np.pi / 8) ** 2 * 2.0,
                    0.5 * np.cos(3 * np.pi / 8) ** 2 * 2.0,
                    0.5 * np.cos(5 * np.pi / 8) ** 2 * 2.0,
                ],
                [
                    1.0 * np.cos(1 * np.pi / 8) ** 2 * 3.0,
                    1.0 * np.cos(7 * np.pi / 8) ** 2 * 3.0,
                    1.0 * np.cos(3 * np.pi / 8) ** 2 * 3.0,
                    1.0 * np.cos(5 * np.pi / 8) ** 2 * 3.0,
                ],
            ]
        )
        vals_expect = vals_expect / (np.pi / 180.0)

        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)

    def test_from_spectrum1d_and_integrate_back_rad(self):
        freq = np.linspace(0.0, 1.0, 50)
        dirs = np.linspace(0.0, 2.0 * np.pi, endpoint=False)
        vals1d = np.random.random(len(freq))
        dirp = np.pi / 4.0

        def spread_fun(f, d):
            return (1.0 / np.pi) * np.cos(d / 2) ** 2

        spectrum = DirectionalSpectrum.from_spectrum1d(
            freq,
            dirs,
            vals1d,
            spread_fun,
            dirp,
            freq_hz=True,
            degrees=False,
            clockwise=False,
            waves_coming_from=False,
        )

        freq_out, vals1d_out = spectrum.spectrum1d(axis=1)

        np.testing.assert_array_almost_equal(freq_out, freq)
        np.testing.assert_array_almost_equal(vals1d_out, vals1d)

    def test_from_spectrum1d_and_integrate_back_deg(self):
        freq = np.linspace(0.0, 1.0, 50)
        dirs = np.linspace(0.0, 360.0, endpoint=False)
        vals1d = np.random.random(len(freq))
        dirp = 45.0

        def spread_fun(f, d):
            return (1.0 / 180.0) * np.cos(np.radians(d / 2)) ** 2

        spectrum = DirectionalSpectrum.from_spectrum1d(
            freq,
            dirs,
            vals1d,
            spread_fun,
            dirp,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=False,
        )

        freq_out, vals1d_out = spectrum.spectrum1d(axis=1)

        np.testing.assert_array_almost_equal(freq_out, freq)
        np.testing.assert_array_almost_equal(vals1d_out, vals1d)

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

    testdata_full_range_dir = [
        ([1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0, 2.0 * np.pi]),
        ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0, 2.0 * np.pi]),
        ([0.0, 1.0, 2.0, 3.0, 2.0 * np.pi], [0.0, 1.0, 2.0, 3.0, 2.0 * np.pi]),
    ]

    @pytest.mark.parametrize("x,expect", testdata_full_range_dir)
    def test_full_range_dir(self, x, expect):
        out = DirectionalSpectrum._full_range_dir(x)
        np.testing.assert_array_almost_equal(out, expect)

    @pytest.mark.parametrize("x,expect", testdata_full_range_dir)
    def test_full_range_dir_float32(self, x, expect):
        x = np.asarray(x, dtype="float32")
        out = DirectionalSpectrum._full_range_dir(x)
        np.testing.assert_array_almost_equal(out, expect)

    def test_var(self):
        y0 = 0.0
        y1 = 2
        a = 7
        b = 6

        y = np.linspace(y0, y1, 20)
        x = np.arange(5, 360, 10)
        v = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        spectrum = DirectionalSpectrum(y, x, v, freq_hz=True, degrees=True)
        var_out = spectrum.var()

        integral_expect = (
            (1.0 / 2.0)
            * (0.0 - 360.0)
            * (y0 - y1)
            * (a * (0.0 + 360.0) + b * (y0 + y1))
        )

        assert var_out == pytest.approx(integral_expect)

    def test_std(self):
        y0 = 0.0
        y1 = 2
        a = 7
        b = 6

        y = np.linspace(y0, y1, 20)
        x = np.arange(5, 360, 10)
        v = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        spectrum = DirectionalSpectrum(y, x, v, freq_hz=True, degrees=True)
        std_out = spectrum.std()

        integral_expect = (
            (1.0 / 2.0)
            * (0.0 - 360.0)
            * (y0 - y1)
            * (a * (0.0 + 360.0) + b * (y0 + y1))
        )

        assert std_out == pytest.approx(np.sqrt(integral_expect))

    def test_spectrum1d_axis1_hz(self):
        yp = np.linspace(0.0, 2.0, 20)
        xp = np.arange(5.0, 360.0, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        f_out, spectrum1d_out = spectrum.spectrum1d(axis=1, freq_hz=True)

        f_expect = yp
        spectrum1d_expect = np.array([360.0 - 0.0] * len(f_expect))

        np.testing.assert_array_almost_equal(f_out, f_expect)
        np.testing.assert_array_almost_equal(spectrum1d_out, spectrum1d_expect)

    def test_spectrum1d_axis1_rads(self):
        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        f_out, spectrum1d_out = spectrum.spectrum1d(axis=1, freq_hz=False)

        f_expect = yp * (2.0 * np.pi)
        spectrum1d_expect = np.array([360.0 - 0.0] * len(f_expect)) / (2.0 * np.pi)

        np.testing.assert_array_almost_equal(f_out, f_expect)
        np.testing.assert_array_almost_equal(spectrum1d_out, spectrum1d_expect)

    def test_spectrum1d_axis0_deg(self):
        f0 = 0.0
        f1 = 2.0
        d0 = 0.0
        d1 = 359.0

        yp = np.linspace(f0, f1, 20)
        xp = np.linspace(d0, d1, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        dir_out, spectrum1d_out = spectrum.spectrum1d(axis=0, degrees=True)

        dir_expect = xp
        spectrum1d_expect = np.array([f1 - f0] * len(dir_expect))

        np.testing.assert_array_almost_equal(dir_out, dir_expect)
        np.testing.assert_array_almost_equal(spectrum1d_out, spectrum1d_expect)

    def test_spectrum1d_axis0_rad(self):
        f0 = 0.0
        f1 = 2.0
        d0 = 0.0
        d1 = 359.0

        yp = np.linspace(f0, f1, 20)
        xp = np.linspace(d0, d1, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        dir_out, spectrum1d_out = spectrum.spectrum1d(axis=0, degrees=False)

        dir_expect = xp * (np.pi / 180.0)
        spectrum1d_expect = np.array([f1 - f0] * len(dir_expect)) / (np.pi / 180.0)

        np.testing.assert_array_almost_equal(dir_out, dir_expect)
        np.testing.assert_array_almost_equal(spectrum1d_out, spectrum1d_expect)

    def test_moment_m0_hz(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(0, freq_hz=True)

        m_expect = (0.0 - 360.0) * (f0 - f1)

        assert m_out == pytest.approx(m_expect)

    def test_moment_m0_hz_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(0, freq_hz=True)

        m_expect = (0.0 - 360.0) * (f0 - f1) + 1j * (0.0 - 360.0) * (f0 - f1)

        assert m_out == pytest.approx(m_expect)

    def test_moment_m0_rads(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(0, freq_hz=False)

        m_expect = (0.0 - 360.0) * (f0 - f1)

        assert m_out == pytest.approx(m_expect)

    def test_moment_m0_rads_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(0, freq_hz=False)

        m_expect = (0.0 - 360.0) * (f0 - f1) + 1j * (0.0 - 360.0) * (f0 - f1)

        assert m_out == pytest.approx(m_expect)

    def test_moment_m1_hz(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(1, freq_hz=True)

        m_expect = (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2)

        assert m_out == pytest.approx(m_expect)

    def test_moment_m1_hz_comlex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(1, freq_hz=True)

        m_expect = (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2) + 1j * (
            1.0 / 2.0
        ) * (0.0 - 360.0) * (f0**2 - f1**2)

        assert m_out == pytest.approx(m_expect)

    def test_moment_m1_rads(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(1, freq_hz=False)

        m_expect = (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2) * (2.0 * np.pi)

        assert m_out == pytest.approx(m_expect)

    def test_moment_m1_rads_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(1, freq_hz=False)

        m_expect = (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2) * (
            2.0 * np.pi
        ) + 1j * (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2) * (2.0 * np.pi)

        assert m_out == pytest.approx(m_expect)

    def test_moment_m2_hz(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(2, freq_hz=True)

        m_expect = (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3)

        # not exactly same due to error in trapezoid for higher order functions
        assert m_out == pytest.approx(m_expect, rel=0.1)

    def test_moment_m2_hz_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(2, freq_hz=True)

        m_expect = (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3) + 1j * (
            1.0 / 3.0
        ) * (0.0 - 360.0) * (f0**3 - f1**3)

        # not exactly same due to error in trapezoid for higher order functions
        assert m_out == pytest.approx(m_expect, rel=0.1)

    def test_moment_m2_rads(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(2, freq_hz=False)

        m_expect = (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3) * (2.0 * np.pi) ** 2

        # not exactly same due to error in trapezoid for higher order functions
        assert m_out == pytest.approx(m_expect, rel=0.1)

    def test_moment_m2_rads_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(2, freq_hz=False)

        m_expect = (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3) * (
            2.0 * np.pi
        ) ** 2 + 1j * (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3) * (
            2.0 * np.pi
        ) ** 2

        # not exactly same due to error in trapezoid for higher order functions
        assert m_out == pytest.approx(m_expect, rel=0.1)

    def test_tz(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        tz_out = spectrum.tz

        m0 = (0.0 - 360.0) * (f0 - f1)
        m2 = (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3)

        tz_expect = np.sqrt(m0 / m2)

        assert tz_out == pytest.approx(tz_expect, rel=0.1)

    def test_tz_raises_complex(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs))) + 1j * np.ones((len(freq), len(dirs)))
        spectrum = DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        with pytest.raises(ValueError):
            _ = spectrum.tz

    def test_tz_raises_neg(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = -np.ones((len(freq), len(dirs)))
        spectrum = DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        with pytest.raises(ValueError):
            _ = spectrum.tz

    def test_from_grid(self):
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

        spectrum = DirectionalSpectrum.from_grid(grid)

        vals_expect = vals.copy()
        vals_expect /= 2.0 * np.pi
        vals_expect /= np.pi / 180.0

        assert isinstance(spectrum, DirectionalSpectrum)
        np.testing.assert_array_almost_equal(spectrum._freq, grid._freq)
        np.testing.assert_array_almost_equal(spectrum._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)
        assert spectrum._clockwise == grid._clockwise
        assert spectrum._waves_coming_from == grid._waves_coming_from
        assert spectrum._freq_hz == grid._freq_hz
        assert spectrum._degrees == grid._degrees

    def test_from_grid_no_value_scaling(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 2.0 * np.pi, 15, endpoint=False)
        vals = np.random.random((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=False,
            degrees=False,
            clockwise=True,
            waves_coming_from=True,
        )

        spectrum = DirectionalSpectrum.from_grid(grid)

        vals_expect = vals.copy()

        assert isinstance(spectrum, DirectionalSpectrum)
        np.testing.assert_array_almost_equal(spectrum._freq, grid._freq)
        np.testing.assert_array_almost_equal(spectrum._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)
        assert spectrum._clockwise == grid._clockwise
        assert spectrum._waves_coming_from == grid._waves_coming_from
        assert spectrum._freq_hz == grid._freq_hz
        assert spectrum._degrees == grid._degrees

    def test_extreme_float(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        T = 360 * 24 * 60.0**2
        q = 0.99
        extreme_out = spectrum.extreme(T, q=q)

        sigma = spectrum.std()
        tz = spectrum.tz

        extreme_expect = sigma * np.sqrt(2.0 * np.log((T / tz) / np.log(1.0 / q)))

        assert extreme_out == pytest.approx(extreme_expect)

    def test_extreme_raises_complex(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs))) + 1j * np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        T = 360 * 24 * 60.0**2
        q = 0.99
        with pytest.raises(ValueError):
            _ = spectrum.extreme(T, q=q)

    def test_extreme_raises_neg(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = -np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        T = 360 * 24 * 60.0**2
        q = 0.99
        with pytest.raises(ValueError):
            _ = spectrum.extreme(T, q=q)

    def test_extreme_list(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        T = 360 * 24 * 60.0**2
        q = [0.1, 0.5, 0.99]
        extreme_out = spectrum.extreme(T, q=q)

        sigma = spectrum.std()
        tz = spectrum.tz

        extreme_expect = [
            sigma * np.sqrt(2.0 * np.log((T / tz) / np.log(1.0 / q[0]))),
            sigma * np.sqrt(2.0 * np.log((T / tz) / np.log(1.0 / q[1]))),
            sigma * np.sqrt(2.0 * np.log((T / tz) / np.log(1.0 / q[2]))),
        ]

        np.testing.assert_array_almost_equal(extreme_out, extreme_expect)

    def test_extreme_mpm(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        T = 360 * 24 * 60.0**2
        extreme_out = spectrum.extreme(T, q=0.37)

        sigma = spectrum.std()
        tz = spectrum.tz

        extreme_expect = sigma * np.sqrt(2.0 * np.log(T / tz))

        assert extreme_out == pytest.approx(extreme_expect, rel=1e-3)

    def test_extreme_absmax(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        T = 360 * 24 * 60.0**2
        q = 0.99
        extreme_out = spectrum.extreme(T, q=q, absmax=True)

        sigma = spectrum.std()
        tz_absmax = spectrum.tz / 2.0

        extreme_expect = sigma * np.sqrt(
            2.0 * np.log((T / tz_absmax) / np.log(1.0 / q))
        )

        assert extreme_out == pytest.approx(extreme_expect)

    def test_reshape(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        grid_reshaped = spectrum.reshape(y, x, freq_hz=True, degrees=True)

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * x
        vals_expect = (
            np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])
            * (180.0 / np.pi)
            / (2.0 * np.pi)
        )

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
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        y_ = (2.0 * np.pi) * y
        x_ = (np.pi / 180.0) * x
        grid_reshaped = spectrum.reshape(y_, x_, freq_hz=False, degrees=False)

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * x
        vals_expect = (
            np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])
            * (180.0 / np.pi)
            / (2.0 * np.pi)
        )

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
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        grid_reshaped = spectrum.reshape(
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
        vals_expect = (
            (vals_real_expect + 1j * vals_imag_expect) * (180.0 / np.pi) / (2.0 * np.pi)
        )

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
        spectrum = DirectionalSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = np.linspace(5.0, 15.0, 10)
        grid_reshaped = spectrum.reshape(
            y, x, freq_hz=True, degrees=True, complex_convert="polar"
        )

        freq_out = grid_reshaped._freq
        dirs_out = grid_reshaped._dirs
        vals_out = grid_reshaped._vals

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * x
        vals_amp_expect = (
            np.array([[a_amp * x_i + b_amp * y_i for x_i in x] for y_i in y])
            * (180.0 / np.pi)
            / (2.0 * np.pi)
        )
        x_, y_ = np.meshgrid(x, y, indexing="ij", sparse=True)
        vals_phase_cos_expect = RGI((xp, yp), np.cos(vp_phase).T)((x_, y_)).T
        vals_phase_sin_expect = RGI((xp, yp), np.sin(vp_phase).T)((x_, y_)).T

        vals_expect = (
            vals_amp_expect
            * (vals_phase_cos_expect + 1j * vals_phase_sin_expect)
            / np.abs(vals_phase_cos_expect + 1j * vals_phase_sin_expect)
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)


class Test_DirectionalBinSpectrum:
    def test__init___hz_deg(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalBinSpectrum(
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
            spectrum._vals, 1.0 / (2.0 * np.pi) * vals_in
        )

    def test__init___hz_rad(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = (np.pi / 180.0) * np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalBinSpectrum(
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
        spectrum = DirectionalBinSpectrum(
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
        np.testing.assert_array_almost_equal(spectrum._vals, vals_in)

    def test__init___rads_rad(self):
        freq_in = (2.0 * np.pi) * np.arange(0.0, 1, 0.1)
        dirs_in = (np.pi / 180.0) * np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalBinSpectrum(
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
            DirectionalBinSpectrum(freq, dirs, values, freq_hz=True, degrees=True)

    def test__init__vals_neg_ok(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals[0, 1] *= -1

        _ = DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__vals_complex_ok(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals = vals + 1j * vals

        _ = DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__raises_freq_neg(self):
        freq = np.arange(-0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        values = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalBinSpectrum(freq, dirs, values, freq_hz=True, degrees=True)

    def test__init__raises_freq_nosort(self):
        freq = np.array([0.5, 0.0, 1.0])
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__raises_dirs_360(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.linspace(0.0, 360.0, 10, endpoint=True)
        vals = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__raises_dirs_2pi(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=True)
        vals = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=False)

    def test__init__raises_dirs_neg(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.linspace(-1.0, 360.0, 10)
        vals = np.random.random(size=(len(freq), len(dirs)))

        with pytest.raises(ValueError):
            DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__repr___(self, directional_bin_spectrum):
        assert str(directional_bin_spectrum) == "DirectionalBinSpectrum"

    def test_grid_rads_rad(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalBinSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.grid(freq_hz=False, degrees=False)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = (np.pi / 180.0) * dirs_in
        vals_expect = 1.0 / (2.0 * np.pi) * vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_grid_hz_rad(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalBinSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.grid(freq_hz=True, degrees=False)

        freq_expect = freq_in
        dirs_expect = (np.pi / 180.0) * dirs_in
        vals_expect = vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_grid_rads_deg(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalBinSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.grid(freq_hz=False, degrees=True)

        freq_expect = 2.0 * np.pi * freq_in
        dirs_expect = dirs_in
        vals_expect = 1.0 / (2.0 * np.pi) * vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_grid__hz_deg(self):
        freq_in = np.arange(0.0, 1, 0.1)
        dirs_in = np.arange(5.0, 360.0, 10.0)
        vals_in = np.random.random(size=(len(freq_in), len(dirs_in)))
        spectrum = DirectionalBinSpectrum(
            freq_in,
            dirs_in,
            vals_in,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_out, dirs_out, vals_out = spectrum.grid(freq_hz=True, degrees=True)

        freq_expect = freq_in
        dirs_expect = dirs_in
        vals_expect = vals_in

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)

    def test_interpolate(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        x = xp

        with pytest.raises(AttributeError):
            _ = spectrum.interpolate(y, x, freq_hz=True, degrees=True)

    def test_var(self):
        y0 = 0.0
        y1 = 2
        a = 7
        b = 6

        y = np.linspace(y0, y1, 20)
        x = np.arange(5, 360, 10)
        v = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        spectrum = DirectionalBinSpectrum(y, x, v, freq_hz=True, degrees=True)
        var_out = spectrum.var()

        integral_expect = (
            (1.0 / 2.0)
            * (0.0 - 360.0)
            / 10
            * (y0 - y1)
            * (a * (0.0 + 360.0) + b * (y0 + y1))
        )

        assert var_out == pytest.approx(integral_expect)

    def test_std(self):
        y0 = 0.0
        y1 = 2
        a = 7
        b = 6

        y = np.linspace(y0, y1, 20)
        x = np.arange(5, 360, 10)
        v = np.array([[a * x_i + b * y_i for x_i in x] for y_i in y])

        spectrum = DirectionalBinSpectrum(y, x, v, freq_hz=True, degrees=True)
        std_out = spectrum.std()

        integral_expect = (
            (1.0 / 2.0)
            * (0.0 - 360.0)
            / 10
            * (y0 - y1)
            * (a * (0.0 + 360.0) + b * (y0 + y1))
        )

        assert std_out == pytest.approx(np.sqrt(integral_expect))

    def test_spectrum1d_axis1_hz(self):
        yp = np.linspace(0.0, 2.0, 20)
        xp = np.arange(5.0, 360.0, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        f_out, spectrum1d_out = spectrum.spectrum1d(axis=1, freq_hz=True)

        f_expect = yp
        spectrum1d_expect = np.array([len(xp)] * len(f_expect))

        np.testing.assert_array_almost_equal(f_out, f_expect)
        np.testing.assert_array_almost_equal(spectrum1d_out, spectrum1d_expect)

    def test_spectrum1d_axis1_rads(self):
        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        f_out, spectrum1d_out = spectrum.spectrum1d(axis=1, freq_hz=False)

        f_expect = yp * (2.0 * np.pi)
        spectrum1d_expect = np.array([len(xp)] * len(f_expect)) / (2.0 * np.pi)

        np.testing.assert_array_almost_equal(f_out, f_expect)
        np.testing.assert_array_almost_equal(spectrum1d_out, spectrum1d_expect)

    def test_spectrum1d_axis0_deg(self):
        f0 = 0.0
        f1 = 2.0
        d0 = 0.0
        d1 = 359.0

        yp = np.linspace(f0, f1, 20)
        xp = np.linspace(d0, d1, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        dir_out, spectrum1d_out = spectrum.spectrum1d(axis=0, degrees=True)

        dir_expect = xp
        spectrum1d_expect = np.array([f1 - f0] * len(dir_expect))

        np.testing.assert_array_almost_equal(dir_out, dir_expect)
        np.testing.assert_array_almost_equal(spectrum1d_out, spectrum1d_expect)

    def test_spectrum1d_axis0_rad(self):
        f0 = 0.0
        f1 = 2.0
        d0 = 0.0
        d1 = 359.0

        yp = np.linspace(f0, f1, 20)
        xp = np.linspace(d0, d1, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        dir_out, spectrum1d_out = spectrum.spectrum1d(axis=0, degrees=False)

        dir_expect = xp * (np.pi / 180.0)
        spectrum1d_expect = np.array([f1 - f0] * len(dir_expect))

        np.testing.assert_array_almost_equal(dir_out, dir_expect)
        np.testing.assert_array_almost_equal(spectrum1d_out, spectrum1d_expect)

    def test_moment_m0_hz(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(0, freq_hz=True)

        m_expect = (0.0 - 360.0) * (f0 - f1) / 10

        assert m_out == pytest.approx(m_expect)

    def test_moment_m0_hz_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(0, freq_hz=True)

        m_expect = ((0.0 - 360.0) * (f0 - f1) + 1j * (0.0 - 360.0) * (f0 - f1)) / 10

        assert m_out == pytest.approx(m_expect)

    def test_moment_m0_rads(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(0, freq_hz=False)

        m_expect = (0.0 - 360.0) * (f0 - f1) / 10

        assert m_out == pytest.approx(m_expect)

    def test_moment_m0_rads_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(0, freq_hz=False)

        m_expect = ((0.0 - 360.0) * (f0 - f1) + 1j * (0.0 - 360.0) * (f0 - f1)) / 10

        assert m_out == pytest.approx(m_expect)

    def test_moment_m1_hz(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(1, freq_hz=True)

        m_expect = (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2) / 10

        assert m_out == pytest.approx(m_expect)

    def test_moment_m1_hz_comlex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(1, freq_hz=True)

        m_expect = (
            (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2)
            + 1j * (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2)
        ) / 10

        assert m_out == pytest.approx(m_expect)

    def test_moment_m1_rads(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(1, freq_hz=False)

        m_expect = ((1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2) * (2.0 * np.pi)) / 10

        assert m_out == pytest.approx(m_expect)

    def test_moment_m1_rads_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(1, freq_hz=False)

        m_expect = (
            (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2) * (2.0 * np.pi)
            + 1j * (1.0 / 2.0) * (0.0 - 360.0) * (f0**2 - f1**2) * (2.0 * np.pi)
        ) / 10

        assert m_out == pytest.approx(m_expect)

    def test_moment_m2_hz(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(2, freq_hz=True)

        m_expect = ((1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3)) / 10

        # not exactly same due to error in trapezoid for higher order functions
        assert m_out == pytest.approx(m_expect, rel=0.1)

    def test_moment_m2_hz_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(2, freq_hz=True)

        m_expect = (
            (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3)
            + 1j * (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3)
        ) / 10

        # not exactly same due to error in trapezoid for higher order functions
        assert m_out == pytest.approx(m_expect, rel=0.1)

    def test_moment_m2_rads(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(2, freq_hz=False)

        m_expect = (
            (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3) * (2.0 * np.pi) ** 2 / 10
        )

        # not exactly same due to error in trapezoid for higher order functions
        assert m_out == pytest.approx(m_expect, rel=0.1)

    def test_moment_m2_rads_complex(self):
        f0 = 0.0
        f1 = 2.0

        yp = np.linspace(f0, f1, 20)
        xp = np.arange(5, 360, 10)
        vp = np.ones((len(yp), len(xp))) + 1j * np.ones((len(yp), len(xp)))
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        m_out = spectrum.moment(2, freq_hz=False)

        m_expect = (
            (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3) * (2.0 * np.pi) ** 2
            + 1j * (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3) * (2.0 * np.pi) ** 2
        ) / 10

        # not exactly same due to error in trapezoid for higher order functions
        assert m_out == pytest.approx(m_expect, rel=0.1)

    def test_tz(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        tz_out = spectrum.tz

        m0 = (0.0 - 360.0) * (f0 - f1)
        m2 = (1.0 / 3.0) * (0.0 - 360.0) * (f0**3 - f1**3)

        tz_expect = np.sqrt(m0 / m2)

        assert tz_out == pytest.approx(tz_expect, rel=0.1)

    def test_tz_raises_complex(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs))) + 1j * np.ones((len(freq), len(dirs)))
        spectrum = DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        with pytest.raises(ValueError):
            _ = spectrum.tz

    def test_tz_raises_neg(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = -np.ones((len(freq), len(dirs)))
        spectrum = DirectionalBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        with pytest.raises(ValueError):
            _ = spectrum.tz

    def test_from_grid(self):
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

        spectrum = DirectionalBinSpectrum.from_grid(grid)

        vals_expect = vals.copy()
        vals_expect /= 2.0 * np.pi

        assert isinstance(spectrum, DirectionalBinSpectrum)
        np.testing.assert_array_almost_equal(spectrum._freq, grid._freq)
        np.testing.assert_array_almost_equal(spectrum._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)
        assert spectrum._clockwise == grid._clockwise
        assert spectrum._waves_coming_from == grid._waves_coming_from
        assert spectrum._freq_hz == grid._freq_hz
        assert spectrum._degrees == grid._degrees

    def test_from_grid_no_value_scaling(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 2.0 * np.pi, 15, endpoint=False)
        vals = np.random.random((10, 15))
        grid = Grid(
            freq,
            dirs,
            vals,
            freq_hz=False,
            degrees=False,
            clockwise=True,
            waves_coming_from=True,
        )

        spectrum = DirectionalBinSpectrum.from_grid(grid)

        vals_expect = vals.copy()

        assert isinstance(spectrum, DirectionalBinSpectrum)
        np.testing.assert_array_almost_equal(spectrum._freq, grid._freq)
        np.testing.assert_array_almost_equal(spectrum._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)
        assert spectrum._clockwise == grid._clockwise
        assert spectrum._waves_coming_from == grid._waves_coming_from
        assert spectrum._freq_hz == grid._freq_hz
        assert spectrum._degrees == grid._degrees

    def test_extreme_float(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalBinSpectrum(
            freq, dirs, vals, freq_hz=True, degrees=True
        )

        T = 360 * 24 * 60.0**2
        q = 0.99
        extreme_out = spectrum.extreme(T, q=q)

        sigma = spectrum.std()
        tz = spectrum.tz

        extreme_expect = sigma * np.sqrt(2.0 * np.log((T / tz) / np.log(1.0 / q)))

        assert extreme_out == pytest.approx(extreme_expect)

    def test_extreme_raises_complex(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs))) + 1j * np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalBinSpectrum(
            freq, dirs, vals, freq_hz=True, degrees=True
        )

        T = 360 * 24 * 60.0**2
        q = 0.99
        with pytest.raises(ValueError):
            _ = spectrum.extreme(T, q=q)

    def test_extreme_raises_neg(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = -np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalBinSpectrum(
            freq, dirs, vals, freq_hz=True, degrees=True
        )

        T = 360 * 24 * 60.0**2
        q = 0.99
        with pytest.raises(ValueError):
            _ = spectrum.extreme(T, q=q)

    def test_extreme_list(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalBinSpectrum(
            freq, dirs, vals, freq_hz=True, degrees=True
        )

        T = 360 * 24 * 60.0**2
        q = [0.1, 0.5, 0.99]
        extreme_out = spectrum.extreme(T, q=q)

        sigma = spectrum.std()
        tz = spectrum.tz

        extreme_expect = [
            sigma * np.sqrt(2.0 * np.log((T / tz) / np.log(1.0 / q[0]))),
            sigma * np.sqrt(2.0 * np.log((T / tz) / np.log(1.0 / q[1]))),
            sigma * np.sqrt(2.0 * np.log((T / tz) / np.log(1.0 / q[2]))),
        ]

        np.testing.assert_array_almost_equal(extreme_out, extreme_expect)

    def test_extreme_mpm(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalBinSpectrum(
            freq, dirs, vals, freq_hz=True, degrees=True
        )

        T = 360 * 24 * 60.0**2
        extreme_out = spectrum.extreme(T, q=0.37)

        sigma = spectrum.std()
        tz = spectrum.tz

        extreme_expect = sigma * np.sqrt(2.0 * np.log(T / tz))

        assert extreme_out == pytest.approx(extreme_expect, rel=1e-3)

    def test_extreme_absmax(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        spectrum = wr.DirectionalBinSpectrum(
            freq, dirs, vals, freq_hz=True, degrees=True
        )

        T = 360 * 24 * 60.0**2
        q = 0.99
        extreme_out = spectrum.extreme(T, q=q, absmax=True)

        sigma = spectrum.std()
        tz_absmax = spectrum.tz / 2.0

        extreme_expect = sigma * np.sqrt(
            2.0 * np.log((T / tz_absmax) / np.log(1.0 / q))
        )

        assert extreme_out == pytest.approx(extreme_expect)

    def test_reshape(self):
        a = 7
        b = 6

        yp = np.linspace(0.0, 2.0, 20)
        xp = np.linspace(0.0, 359.0, 10)
        vp = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in yp])
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        grid_reshaped = spectrum.reshape(y, freq_hz=True)

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * xp
        vals_expect = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in y]) / (
            2.0 * np.pi
        )

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
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        y_ = (2.0 * np.pi) * y
        grid_reshaped = spectrum.reshape(y_, freq_hz=False)

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * xp
        vals_expect = np.array([[a * x_i + b * y_i for x_i in xp] for y_i in y]) / (
            2.0 * np.pi
        )

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
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        grid_reshaped = spectrum.reshape(y, freq_hz=True, complex_convert="rectangular")

        freq_out = grid_reshaped._freq
        dirs_out = grid_reshaped._dirs
        vals_out = grid_reshaped._vals

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * xp
        vals_real_expect = np.array(
            [[a_real * x_i + b_real * y_i for x_i in xp] for y_i in y]
        )
        vals_imag_expect = np.array(
            [[a_imag * x_i + b_imag * y_i for x_i in xp] for y_i in y]
        )
        vals_expect = (vals_real_expect + 1j * vals_imag_expect) / (2.0 * np.pi)

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
        spectrum = DirectionalBinSpectrum(yp, xp, vp, freq_hz=True, degrees=True)

        y = np.linspace(0.5, 1.0, 20)
        grid_reshaped = spectrum.reshape(y, freq_hz=True, complex_convert="polar")

        freq_out = grid_reshaped._freq
        dirs_out = grid_reshaped._dirs
        vals_out = grid_reshaped._vals

        freq_expect = (2.0 * np.pi) * y
        dirs_expect = (np.pi / 180.0) * xp
        vals_amp_expect = np.array(
            [[a_amp * x_i + b_amp * y_i for x_i in xp] for y_i in y]
        ) / (2.0 * np.pi)
        x_, y_ = np.meshgrid(xp, y, indexing="ij", sparse=True)
        vals_phase_cos_expect = RGI((xp, yp), np.cos(vp_phase).T)((x_, y_)).T
        vals_phase_sin_expect = RGI((xp, yp), np.sin(vp_phase).T)((x_, y_)).T

        vals_expect = (
            vals_amp_expect
            * (vals_phase_cos_expect + 1j * vals_phase_sin_expect)
            / np.abs(vals_phase_cos_expect + 1j * vals_phase_sin_expect)
        )

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(dirs_out, dirs_expect)
        np.testing.assert_array_almost_equal(vals_out, vals_expect)


class Test_WaveSpectrum:
    def test__init__(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.random.random((10, 15))
        wave = WaveSpectrum(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_expect = 2.0 * np.pi * freq
        dirs_expect = (np.pi / 180.0) * dirs
        vals_expect = vals / (2.0 * np.pi * (np.pi / 180.0))

        assert isinstance(wave, Grid)
        assert isinstance(wave, DirectionalSpectrum)
        np.testing.assert_array_almost_equal(wave._freq, freq_expect)
        np.testing.assert_array_almost_equal(wave._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(wave._vals, vals_expect)
        assert wave._clockwise is True
        assert wave._waves_coming_from is True
        assert wave._freq_hz is True
        assert wave._degrees is True

    def test_from_grid(self):
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

        spectrum = WaveSpectrum.from_grid(grid)

        vals_expect = vals.copy()
        vals_expect /= 2.0 * np.pi
        vals_expect /= np.pi / 180.0

        assert isinstance(spectrum, WaveSpectrum)
        np.testing.assert_array_almost_equal(spectrum._freq, grid._freq)
        np.testing.assert_array_almost_equal(spectrum._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)
        assert spectrum._clockwise == grid._clockwise
        assert spectrum._waves_coming_from == grid._waves_coming_from
        assert spectrum._freq_hz == grid._freq_hz
        assert spectrum._degrees == grid._degrees

    def test__init__raises_vals_neg(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals[0, 1] *= -1

        with pytest.raises(ValueError):
            WaveSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__raises_vals_complex(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals = vals + 1j * vals

        with pytest.raises(ValueError):
            WaveSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__repr__(self, wave):
        assert str(wave) == "WaveSpectrum"

    def test_conjugate_raises(self, wave):
        with pytest.raises(AttributeError):
            wave.conjugate()

    def test_real_raises(self, wave):
        with pytest.raises(AttributeError):
            wave.real

    def test_imag_raises(self, wave):
        with pytest.raises(AttributeError):
            wave.imag

    def test_hs(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        wave = WaveSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        hs_out = wave.hs
        hs_expect = 4.0 * np.sqrt((0.0 - 360.0) * (f0 - f1))

        assert hs_out == pytest.approx(hs_expect)

    def test_tp_hz(self):
        freq = np.linspace(0, 2, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))

        idx_dirs_max = 4
        idx_freq_max = 10
        vals[idx_freq_max, idx_dirs_max] = 2.0

        wave = WaveSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        tp_out = wave.tp

        fp_expect = freq[idx_freq_max]
        tp_expect = 1.0 / fp_expect

        assert tp_out == tp_expect

    def test_tp_rads(self):
        freq = np.linspace(0, 2, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))

        idx_dirs_max = 4
        idx_freq_max = 10
        vals[idx_freq_max, idx_dirs_max] = 2.0

        wave = WaveSpectrum(freq, dirs, vals, freq_hz=False, degrees=True)

        tp_out = wave.tp

        fp_expect = freq[idx_freq_max] / (2.0 * np.pi)
        tp_expect = 1.0 / fp_expect

        assert tp_out == tp_expect

    testdata_mean_direction = [
        (0.0, np.pi / 2, np.pi / 4),
        (np.pi / 2.0, np.pi, 3.0 * np.pi / 4),
        (np.pi, 3.0 * np.pi / 2.0, 5.0 * np.pi / 4),
        (3.0 * np.pi / 2.0, 2.0 * np.pi, 7.0 * np.pi / 4),
        (np.pi / 2.0, 3.0 * np.pi / 2.0, np.pi),
    ]

    @pytest.mark.parametrize("d0,d1,expect", testdata_mean_direction)
    def test_mean_direction(self, d0, d1, expect):
        dirs = np.linspace(0, 2.0 * np.pi, 100)
        spectrum1d = np.zeros_like(dirs)

        dirs_mask = (dirs >= d0) & (dirs <= d1)
        spectrum1d[dirs_mask] = 1.0

        out = WaveSpectrum._mean_direction(dirs, spectrum1d)

        assert out == pytest.approx(expect, rel=0.1)

    @pytest.mark.parametrize("d0,d1,expect", testdata_mean_direction)
    def test_dirp_deg(self, d0, d1, expect):
        freq = np.linspace(0, 2, 20)
        dirs = np.linspace(0, 2.0 * np.pi - 1e-3, 100)
        vals = np.zeros((len(freq), len(dirs)))

        dir_mask = (dirs >= d0) & (dirs <= d1)
        idx_freq_max = 10
        vals[idx_freq_max, dir_mask] = 1.0

        wave = WaveSpectrum(freq, dirs, vals, freq_hz=True, degrees=False)

        dirp_out = wave.dirp(degrees=True)
        dirp_expect = (180.0 / np.pi) * expect

        assert dirp_out == pytest.approx(dirp_expect, rel=0.1)

    @pytest.mark.parametrize("d0,d1,expect", testdata_mean_direction)
    def test_dirp_rad(self, d0, d1, expect):
        freq = np.linspace(0, 2, 20)
        dirs = np.linspace(0, 2.0 * np.pi - 1e-3, 100)
        vals = np.zeros((len(freq), len(dirs)))

        dirs_mask = (dirs >= d0) & (dirs <= d1)
        idx_freq_max = 10
        vals[idx_freq_max, dirs_mask] = 1.0

        wave = WaveSpectrum(freq, dirs, vals, freq_hz=True, degrees=False)

        dirp_out = wave.dirp(degrees=False)
        dirp_expect = expect

        assert dirp_out == pytest.approx(dirp_expect, rel=0.1)

    @pytest.mark.parametrize("d0,d1,expect", testdata_mean_direction)
    def test_dirm_deg(self, d0, d1, expect):
        freq = np.linspace(0, 2, 20)
        dirs = np.linspace(0, 2.0 * np.pi - 1e-3, 100)
        vals = np.zeros((len(freq), len(dirs)))

        dirs_mask = (dirs >= d0) & (dirs <= d1)
        vals[:, dirs_mask] = 1.0

        wave = WaveSpectrum(freq, dirs, vals, freq_hz=True, degrees=False)

        dirm_out = wave.dirm(degrees=True)
        dirm_expect = (180.0 / np.pi) * expect

        assert dirm_out == pytest.approx(dirm_expect, rel=0.1)

    @pytest.mark.parametrize("d0,d1,expect", testdata_mean_direction)
    def test_dirm_rad(self, d0, d1, expect):
        freq = np.linspace(0, 2, 20)
        dirs = np.linspace(0, 2.0 * np.pi - 1e-3, 100)
        vals = np.zeros((len(freq), len(dirs)))

        dirs_mask = (dirs >= d0) & (dirs <= d1)
        vals[:, dirs_mask] = 1.0

        wave = WaveSpectrum(freq, dirs, vals, freq_hz=True, degrees=False)

        dirm_out = wave.dirm(degrees=False)
        dirm_expect = expect

        assert dirm_out == pytest.approx(dirm_expect, rel=0.1)


class Test_WaveBinSpectrum:
    def test__init__(self):
        freq = np.linspace(0, 1.0, 10)
        dirs = np.linspace(0, 360.0, 15, endpoint=False)
        vals = np.random.random((10, 15))
        wave = WaveBinSpectrum(
            freq,
            dirs,
            vals,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        freq_expect = 2.0 * np.pi * freq
        dirs_expect = (np.pi / 180.0) * dirs
        vals_expect = vals / (2.0 * np.pi)

        assert isinstance(wave, Grid)
        assert isinstance(wave, DirectionalBinSpectrum)
        np.testing.assert_array_almost_equal(wave._freq, freq_expect)
        np.testing.assert_array_almost_equal(wave._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(wave._vals, vals_expect)
        assert wave._clockwise is True
        assert wave._waves_coming_from is True
        assert wave._freq_hz is True
        assert wave._degrees is True

    def test_from_grid(self):
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

        spectrum = WaveBinSpectrum.from_grid(grid)

        vals_expect = vals.copy()
        vals_expect /= 2.0 * np.pi

        assert isinstance(spectrum, WaveBinSpectrum)
        np.testing.assert_array_almost_equal(spectrum._freq, grid._freq)
        np.testing.assert_array_almost_equal(spectrum._dirs, grid._dirs)
        np.testing.assert_array_almost_equal(spectrum._vals, vals_expect)
        assert spectrum._clockwise == grid._clockwise
        assert spectrum._waves_coming_from == grid._waves_coming_from
        assert spectrum._freq_hz == grid._freq_hz
        assert spectrum._degrees == grid._degrees

    def test__init__raises_vals_neg(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals[0, 1] *= -1

        with pytest.raises(ValueError):
            WaveBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__init__raises_vals_complex(self):
        freq = np.arange(0.05, 1, 0.1)
        dirs = np.arange(5.0, 360.0, 10.0)
        vals = np.random.random(size=(len(freq), len(dirs)))
        vals = vals + 1j * vals

        with pytest.raises(ValueError):
            WaveBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

    def test__repr__(self, wavebin):
        assert str(wavebin) == "WaveBinSpectrum"

    def test_conjugate_raises(self, wavebin):
        with pytest.raises(AttributeError):
            wavebin.conjugate()

    def test_real_raises(self, wavebin):
        with pytest.raises(AttributeError):
            wavebin.real

    def test_imag_raises(self, wavebin):
        with pytest.raises(AttributeError):
            wavebin.imag

    def test_hs(self):
        f0 = 0.0
        f1 = 2.0

        freq = np.linspace(f0, f1, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))
        wave = WaveBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        hs_out = wave.hs
        var_expect = 1.0 * (f1 - f0) * len(dirs)
        hs_expect = 4.0 * np.sqrt(var_expect)

        assert hs_out == pytest.approx(hs_expect)

    def test_tp_hz(self):
        freq = np.linspace(0, 2, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))

        idx_dirs_max = 4
        idx_freq_max = 10
        vals[idx_freq_max, idx_dirs_max] = 2.0

        wave = WaveBinSpectrum(freq, dirs, vals, freq_hz=True, degrees=True)

        tp_out = wave.tp

        fp_expect = freq[idx_freq_max]
        tp_expect = 1.0 / fp_expect

        assert tp_out == tp_expect

    def test_tp_rads(self):
        freq = np.linspace(0, 2, 20)
        dirs = np.arange(5, 360, 10)
        vals = np.ones((len(freq), len(dirs)))

        idx_dirs_max = 4
        idx_freq_max = 10
        vals[idx_freq_max, idx_dirs_max] = 2.0

        wave = WaveBinSpectrum(freq, dirs, vals, freq_hz=False, degrees=True)

        tp_out = wave.tp

        fp_expect = freq[idx_freq_max] / (2.0 * np.pi)
        tp_expect = 1.0 / fp_expect

        assert tp_out == tp_expect

    testdata_mean_direction = [
        ([0.0, np.pi / 2.0, np.pi], [1.0, 1.0, 1.0], np.pi / 2.0),
        ([0.0, np.pi / 2.0, np.pi], [0.0, 0.0, 1.0], np.pi),
        ([0.0, np.pi / 2.0, np.pi], [0.0, 1.0, 0.0], np.pi / 2.0),
        ([0.0, np.pi / 2.0, np.pi], [1.0, 0.0, 0.0], 0.0),
        (
            [np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi - 1e-8],
            [1.0, 1.0, 1.0],
            3.0 * np.pi / 2.0,
        ),
    ]

    @pytest.mark.parametrize("d,s,mean_dir_rad", testdata_mean_direction)
    def test__mean_direction(self, d, s, mean_dir_rad):
        d = np.asarray(d)
        s = np.asarray(s)
        meandir_out = WaveBinSpectrum._mean_direction(d, s)
        assert meandir_out == pytest.approx(mean_dir_rad)

    @pytest.mark.parametrize("d,s,mean_dir_rad", testdata_mean_direction)
    def test_dirp_deg(self, d, s, mean_dir_rad):
        d = np.asarray(d)
        s = np.asarray(s)
        f = np.linspace(0, 2, 20)
        v = np.zeros((len(f), len(d)))

        idx_freq_max = 10
        v[idx_freq_max, :] = s

        wave = WaveBinSpectrum(f, d, v, freq_hz=True, degrees=False)

        dirp_out = wave.dirp(degrees=True)
        dirp_expect = (180.0 / np.pi) * mean_dir_rad

        assert dirp_out == pytest.approx(dirp_expect)

    @pytest.mark.parametrize("d,s,mean_dir_rad", testdata_mean_direction)
    def test_dirp_rad(self, d, s, mean_dir_rad):
        d = np.asarray(d)
        s = np.asarray(s)
        f = np.linspace(0, 2, 20)
        v = np.zeros((len(f), len(d)))

        idx_freq_max = 10
        v[idx_freq_max, :] = s

        wave = WaveBinSpectrum(f, d, v, freq_hz=True, degrees=False)

        dirp_out = wave.dirp(degrees=False)
        dirp_expect = mean_dir_rad

        assert dirp_out == pytest.approx(dirp_expect)

    @pytest.mark.parametrize("d,s,mean_dir_rad", testdata_mean_direction)
    def test_dirm_deg(self, d, s, mean_dir_rad):
        d = np.asarray(d)
        s = np.asarray(s)
        f = np.linspace(0, 2, 20)
        v = np.tile(s, (len(f), 1))

        wave = WaveBinSpectrum(f, d, v, freq_hz=True, degrees=False)

        dirm_out = wave.dirm(degrees=True)
        dirm_expect = (180.0 / np.pi) * mean_dir_rad

        assert dirm_out == pytest.approx(dirm_expect)

    @pytest.mark.parametrize("d,s,mean_dir_rad", testdata_mean_direction)
    def test_dirm_rad(self, d, s, mean_dir_rad):
        d = np.asarray(d)
        s = np.asarray(s)
        f = np.linspace(0, 2, 20)
        v = np.tile(s, (len(f), 1))

        wave = WaveBinSpectrum(f, d, v, freq_hz=True, degrees=False)

        dirm_out = wave.dirm(degrees=False)
        dirm_expect = mean_dir_rad

        assert dirm_out == pytest.approx(dirm_expect)


class Test_calculate_response:
    @pytest.fixture
    def rao(self):
        freq = np.linspace(0.01, 2.0 * np.pi, 10, endpoint=False)  # rad/s
        dirs = np.linspace(0.0, 2.0 * np.pi, 15, endpoint=False)  # rad

        # random function: 2.0 * f + 3.0 * theta + 4.0
        vals_amp = 2.0 * freq[:, np.newaxis] + 3.0 * dirs[np.newaxis, :] + 4.0
        vals_phase = np.linspace(0, 2 * np.pi, len(freq) * len(dirs)).reshape(
            len(freq), len(dirs)
        )

        rao = RAO.from_amp_phase(
            freq,
            dirs,
            vals_amp,
            vals_phase,
            freq_hz=False,
            degrees=False,
            waves_coming_from=True,
            clockwise=False,
        )
        return rao

    @pytest.fixture
    def wave(self):
        freq = np.linspace(0.005, 2.0 * np.pi, 10, endpoint=False)  # rad/s
        dirs = np.linspace(0.5, 2.0 * np.pi, 15, endpoint=False)  # rad

        # random function: f + sqrt(theta) + 7.0
        vals = freq[:, np.newaxis] ** 2 + np.sqrt(dirs[np.newaxis, :]) + 7.0

        wave = WaveSpectrum(
            freq,
            dirs,
            vals,
            freq_hz=False,
            degrees=False,
            waves_coming_from=False,
            clockwise=True,
        )
        return wave

    @pytest.fixture
    def wavebin(self):
        freq = np.linspace(0.005, 2.0 * np.pi, 10, endpoint=False)  # rad/s
        dirs = np.linspace(0.5, 2.0 * np.pi, 15, endpoint=False)  # rad

        # random function: f + sqrt(theta) + 7.0
        vals = freq[:, np.newaxis] ** 2 + np.sqrt(dirs[np.newaxis, :]) + 7.0

        wave = WaveBinSpectrum(
            freq,
            dirs,
            vals,
            freq_hz=False,
            degrees=False,
            waves_coming_from=False,
            clockwise=True,
        )
        return wave

    def test_calculate_response(self, rao, wave):
        response = calculate_response(rao, wave, np.radians(45.0))

        # Expected response
        wave_body = wave.rotate(45.0, degrees=True)
        wave_body.set_wave_convention(waves_coming_from=True, clockwise=False)
        freq_expect, dirs_expect = wave_body._freq, wave_body._dirs
        rao_squared_expect = (rao * rao.conjugate()).real
        rao_squared_expect = rao_squared_expect.reshape(
            freq_expect, dirs_expect, freq_hz=False, degrees=False
        )
        response_expect = wr.multiply(
            rao_squared_expect, wave_body, "DirectionalSpectrum"
        )

        assert isinstance(response, wr.DirectionalSpectrum)
        assert response._clockwise is False
        assert response._waves_coming_from is True
        assert response._freq_hz is False
        assert response._degrees is False
        np.testing.assert_allclose(response._freq, response_expect._freq)
        np.testing.assert_allclose(response._dirs, response_expect._dirs)
        np.testing.assert_allclose(response._vals, response_expect._vals)

    def test_calculate_response_bin(self, rao, wavebin):
        response = calculate_response(rao, wavebin, np.radians(45.0))

        # Expected response
        wave_body = wavebin.rotate(45.0, degrees=True)
        wave_body.set_wave_convention(waves_coming_from=True, clockwise=False)
        freq_expect, dirs_expect = wave_body._freq, wave_body._dirs
        rao_squared_expect = (rao * rao.conjugate()).real
        rao_squared_expect = rao_squared_expect.reshape(
            freq_expect, dirs_expect, freq_hz=False, degrees=False
        )
        response_expect = wr.multiply(
            rao_squared_expect, wave_body, "DirectionalBinSpectrum"
        )

        assert isinstance(response, wr.DirectionalBinSpectrum)
        assert response._clockwise is False
        assert response._waves_coming_from is True
        assert response._freq_hz is False
        assert response._degrees is False
        np.testing.assert_allclose(response._freq, response_expect._freq)
        np.testing.assert_allclose(response._dirs, response_expect._dirs)
        np.testing.assert_allclose(response._vals, response_expect._vals)

    def test_calculate_response_heading_degrees(self, rao, wave):
        response = calculate_response(rao, wave, 45, heading_degrees=True)

        # Expected response
        wave_body = wave.rotate(45.0, degrees=True)
        wave_body.set_wave_convention(waves_coming_from=True, clockwise=False)
        freq_expect, dirs_expect = wave_body._freq, wave_body._dirs
        rao_squared_expect = (rao * rao.conjugate()).real
        rao_squared_expect = rao_squared_expect.reshape(
            freq_expect, dirs_expect, freq_hz=False, degrees=False
        )
        response_expect = wr.multiply(
            rao_squared_expect, wave_body, "DirectionalSpectrum"
        )

        np.testing.assert_allclose(response._freq, response_expect._freq)
        np.testing.assert_allclose(response._dirs, response_expect._dirs)
        np.testing.assert_allclose(response._vals, response_expect._vals)

    def test_calculate_response_heading_radians(self, rao, wave):
        response = calculate_response(rao, wave, np.radians(45), heading_degrees=False)

        # Expected response
        wave_body = wave.rotate(45.0, degrees=True)
        wave_body.set_wave_convention(waves_coming_from=True, clockwise=False)
        freq_expect, dirs_expect = wave_body._freq, wave_body._dirs
        rao_squared_expect = (rao * rao.conjugate()).real
        rao_squared_expect = rao_squared_expect.reshape(
            freq_expect, dirs_expect, freq_hz=False, degrees=False
        )
        response_expect = wr.multiply(
            rao_squared_expect, wave_body, "DirectionalSpectrum"
        )

        np.testing.assert_allclose(response._freq, response_expect._freq)
        np.testing.assert_allclose(response._dirs, response_expect._dirs)
        np.testing.assert_allclose(response._vals, response_expect._vals)

    def test_calculate_response_reshape_rao_squared(self, rao, wave):
        response = calculate_response(rao, wave, np.radians(45), reshape="rao_squared")

        # Expected response
        wave_body = wave.rotate(45.0, degrees=True)
        wave_body.set_wave_convention(waves_coming_from=True, clockwise=False)
        freq_expect, dirs_expect = wave_body._freq, wave_body._dirs
        rao_squared_expect = (rao * rao.conjugate()).real
        rao_squared_expect = rao_squared_expect.reshape(
            freq_expect, dirs_expect, freq_hz=False, degrees=False
        )  # reshape squared RAO
        response_expect = wr.multiply(
            rao_squared_expect, wave_body, "DirectionalSpectrum"
        )

        np.testing.assert_allclose(response._freq, response_expect._freq)
        np.testing.assert_allclose(response._dirs, response_expect._dirs)
        np.testing.assert_allclose(response._vals, response_expect._vals)

    def test_calculate_response_reshape_rao(self, rao, wave):
        response = calculate_response(rao, wave, np.radians(45), reshape="rao")

        # Expected response
        wave_body = wave.rotate(45.0, degrees=True)
        wave_body.set_wave_convention(waves_coming_from=True, clockwise=False)
        freq_expect, dirs_expect = wave_body._freq, wave_body._dirs
        rao_expect = rao.reshape(
            freq_expect, dirs_expect, freq_hz=False, degrees=False
        )  # reshape RAO
        rao_squared_expect = (rao_expect * rao_expect.conjugate()).real
        response_expect = wr.multiply(
            rao_squared_expect, wave_body, "DirectionalSpectrum"
        )

        np.testing.assert_allclose(response._freq, response_expect._freq)
        np.testing.assert_allclose(response._dirs, response_expect._dirs)
        np.testing.assert_allclose(response._vals, response_expect._vals)

    def test_calculate_response_reshape_raises(self, rao, wave):
        with pytest.raises(ValueError):
            calculate_response(rao, wave, np.radians(45), reshape="invalid-value")

    def test_calculate_response_raises_coord_freq(self, rao, wave):
        # TODO: Deprecated functionality. Remove test in future.
        with pytest.raises(ValueError):
            calculate_response(rao, wave, 0.0, coord_freq="invalid-value")

    def test_calculate_response_raises_coord_dirs(self, rao, wave):
        # TODO: Deprecated functionality. Remove test in future.
        with pytest.raises(ValueError):
            calculate_response(rao, wave, 0.0, coord_freq="invalid-value")

    def test_calculate_response_coord_wave(self):
        # TODO: Deprecated functionality. Remove test in future.
        freq_rao = np.array([0.0, 0.5, 1.0])
        dirs_rao = np.array([45.0, 135.0, 225.0, 315.0])
        vals_rao = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 0.0j, 0.0 + 1.0j],
            ]
        )  # all amplitudes are 1
        rao = RAO(
            freq_rao,
            dirs_rao,
            vals_rao,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=False,
        )

        freq_wave = np.array([0.0, 0.3, 0.6, 0.9])
        dirs_wave = np.array([0.0, 90.0, 180.0, 270.0, 359.0])
        vals_wave = np.ones((len(freq_wave), len(dirs_wave)))
        vals_wave = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
            ]
        )
        wave = WaveSpectrum(
            freq_wave,
            dirs_wave,
            vals_wave,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        response = calculate_response(
            rao, wave, 0.0, coord_freq="wave", coord_dirs="wave"
        )

        assert response._clockwise == rao._clockwise
        assert response._waves_coming_from == rao._waves_coming_from

        response.set_wave_convention(**wave.wave_convention)

        freq_expect = wave._freq
        dirs_expect = wave._dirs
        vals_expect = (
            1.0
            / (2.0 * np.pi * np.pi / 180.0)
            * np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0, 20.0],
                ]
            )
        )

        assert isinstance(response, DirectionalSpectrum)
        assert response._freq_hz is False
        assert response._degrees is False
        np.testing.assert_array_almost_equal(response._freq, freq_expect)
        np.testing.assert_array_almost_equal(response._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(response._vals, vals_expect)

    def test_calculate_response_coord_rao(self):
        # TODO: Deprecated functionality. Remove test in future.
        freq_rao = np.array([0.0, 0.5, 1.0])
        dirs_rao = np.array([45.0, 135.0, 225.0, 315.0])
        vals_rao = np.array(
            [
                [1.0 + 0.0j, 0.0 + 2.0j, 3.0 + 0.0j, 0.0 + 4.0j],
                [5.0 + 0.0j, 0.0 + 6.0j, 7.0 + 0.0j, 0.0 + 8.0j],
                [9.0 + 0.0j, 0.0 + 10.0j, 11.0 + 0.0j, 0.0 + 12.0j],
            ]
        )  # all amplitudes are 1
        rao = RAO(
            freq_rao,
            dirs_rao,
            vals_rao,
            freq_hz=True,
            degrees=True,
            clockwise=False,
            waves_coming_from=False,
        )

        freq_wave = np.array([0.0, 0.3, 0.6, 0.9])  # extrapolation needed
        dirs_wave = np.array([0.0, 90.0, 180.0, 270.0, 359.0])
        vals_wave = np.ones((len(freq_wave), len(dirs_wave)))
        wave = WaveSpectrum(
            freq_wave,
            dirs_wave,
            vals_wave,
            freq_hz=True,
            degrees=True,
            clockwise=True,
            waves_coming_from=True,
        )

        response = calculate_response(
            rao, wave, 0.0, coord_freq="rao", coord_dirs="rao"
        )

        freq_expect = rao._freq
        dirs_expect = rao._dirs
        vals_expect = (
            1.0
            / (2.0 * np.pi * np.pi / 180.0)
            * np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [0.0, 0.0, 0.0, 0.0],  # extrapolated
                ]
            )
            ** 2
        )

        assert isinstance(response, DirectionalSpectrum)
        assert response._freq_hz is False
        assert response._degrees is False
        np.testing.assert_array_almost_equal(response._freq, freq_expect)
        np.testing.assert_array_almost_equal(response._dirs, dirs_expect)
        np.testing.assert_array_almost_equal(response._vals, vals_expect)
        assert response._clockwise == rao._clockwise
        assert response._waves_coming_from == rao._waves_coming_from


class Test__check_is_similar:
    def test_check_is_similar(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0, 359.0])

        vals_a = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        vals_b = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        vals_c = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j, 1.0 + 1.0j],
                [0.0 + 1.0j, 1.0 + 0.0j, 1.0 + 1.0j],
            ]
        )

        grid_a = Grid(freq, dirs, vals_a, degrees=True)
        grid_b = Grid(freq, dirs, vals_b, degrees=True)
        grid_c = Grid(freq, dirs, vals_c, degrees=True)
        _check_is_similar(grid_a, grid_b, grid_c)

        grid_a = Grid(freq, dirs, vals_a, degrees=True)
        grid_b = Grid(freq, dirs, vals_b, degrees=True)
        grid_c = Grid(freq, dirs, vals_c, degrees=True)
        _check_is_similar(grid_a, grid_b, grid_c, exact_type=True)

        grid_a = Grid(freq, dirs, vals_a, degrees=True)
        grid_b = Grid(freq, dirs, vals_b, degrees=True)
        grid_c = RAO(freq, dirs, vals_c, degrees=True)
        _check_is_similar(grid_a, grid_b, grid_c, exact_type=False)

    def test_raises_type(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0, 359.0])

        vals_a = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        vals_b = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        vals_c = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j, 1.0 + 1.0j],
                [0.0 + 1.0j, 1.0 + 0.0j, 1.0 + 1.0j],
            ]
        )

        grid_a = Grid(freq, dirs, vals_a, degrees=True)
        grid_b = Grid(freq, dirs, vals_b, degrees=True)
        grid_c = RAO(freq, dirs, vals_c, degrees=True)
        with pytest.raises(TypeError):
            _check_is_similar(grid_a, grid_b, grid_c, exact_type=True)

    def test_raises_exact_type(self, grid, wave):
        with pytest.raises(TypeError):
            _check_is_similar(wave, grid)

        with pytest.raises(TypeError):
            _check_is_similar(grid, wave)

    def test_raises_convention(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0, 359.0])

        vals_a = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        vals_b = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        vals_c = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j, 1.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j, 1.0 + 1.0j],
                [0.0 + 1.0j, 1.0 + 0.0j, 1.0 + 1.0j],
            ]
        )

        grid_a = Grid(freq, dirs, vals_a, degrees=True)
        grid_b = Grid(freq, dirs, vals_b, degrees=True)
        grid_c = Grid(freq, dirs, vals_c, degrees=True)

        grid_a.set_wave_convention(clockwise=True)
        grid_b.set_wave_convention(clockwise=False)

        with pytest.raises(ValueError):
            _check_is_similar(grid_a, grid_b, grid_c, exact_type=True)

    def test_raises_freq(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0, 359.0])

        vals_a = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        vals_b = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        grid_a = Grid(freq, dirs, vals_a, degrees=True)
        grid_b = Grid(freq, dirs, vals_b, degrees=True)

        freq_c = np.array([0.0, 0.5])
        dirs_c = np.array([0.0, 180.0, 359.0])
        vals_c = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        grid_c = Grid(freq_c, dirs_c, vals_c, degrees=True)

        with pytest.raises(ValueError):
            _check_is_similar(grid_a, grid_b, grid_c)

    def test_raises_dirs(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0, 359.0])

        vals_a = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        vals_b = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        grid_a = Grid(freq, dirs, vals_a, degrees=True)
        grid_b = Grid(freq, dirs, vals_b, degrees=True)

        freq_c = np.array([0.0, 0.5, 1.0])
        dirs_c = np.array([0.0, 180.0])
        vals_c = np.array(
            [
                [1.0, 2.0],
                [4.0, 5.0],
                [7.0, 8.0],
            ]
        )
        grid_c = Grid(freq_c, dirs_c, vals_c, degrees=True)

        with pytest.raises(ValueError):
            _check_is_similar(grid_a, grid_b, grid_c)


class Test_CosineFullSpreading:
    def test__init__(self):
        spreading = CosineFullSpreading(123, degrees=True)
        assert isinstance(spreading, wr._core.BaseSpreading)
        assert spreading._s == 123
        assert spreading._degrees is True

    params__call__degrees = [
        [0, 0, 0, 0.002777777777777778],
        [1, 0, 0, 0.002777777777777778],
        [0, 360, 0, 0.002777777777777778],
        [0, -360, 0, 0.002777777777777778],
        [0, 45, 0, 0.002777777777777778],
        [0, -45, 0, 0.002777777777777778],
        [0, 45 + 2 * 360, 0, 0.002777777777777778],
        [0, 0, 1, 0.005555555555555556],
        [1, 0, 1, 0.005555555555555556],
        [0, 360, 1, 0.005555555555555556],
        [0, -360, 1, 0.005555555555555556],
        [0, 45, 1, 0.004741963281073743],
        [0, -45, 1, 0.004741963281073743],
        [0, 45 + 2 * 360, 1, 0.004741963281073743],
        [0, 0, 2, 0.007407407407407408],
        [1, 0, 2, 0.007407407407407408],
        [0, 360, 2, 0.007407407407407408],
        [0, -360, 2, 0.007407407407407408],
        [0, 45, 2, 0.005396691782172398],
        [0, -45, 2, 0.005396691782172398],
        [0, 45 + 2 * 360, 2, 0.005396691782172398],
        [0, 359, 0.5, 0.004363156988011003],
    ]

    @pytest.mark.parametrize("f,d,s,spread_expect", params__call__degrees)
    def test__call__degrees(self, f, d, s, spread_expect):
        spreading = CosineFullSpreading(s, degrees=True)
        assert spreading(f, d) == pytest.approx(spread_expect)

    params__call__radians = [
        [0, 0, 0, 0.15915494309189535],
        [1, 0, 0, 0.15915494309189535],
        [0, 2.0 * np.pi, 0, 0.15915494309189535],
        [0, -2.0 * np.pi, 0, 0.15915494309189535],
        [0, np.pi / 4, 0, 0.15915494309189535],
        [0, -np.pi / 4, 0, 0.15915494309189535],
        [0, np.pi / 4 + 2 * 360, 0, 0.15915494309189535],
        [0, 0, 1, 0.3183098861837907],
        [1, 0, 1, 0.3183098861837907],
        [0, 2.0 * np.pi, 1, 0.3183098861837907],
        [0, -2.0 * np.pi, 1, 0.3183098861837907],
        [0, np.pi / 4, 1, 0.2716944826115336],
        [0, -np.pi / 4, 1, 0.2716944826115336],
        [0, np.pi / 4 + 2 * 2.0 * np.pi, 1, 0.2716944826115336],
        [0, 0, 2, 0.4244131815783876],
        [1, 0, 2, 0.4244131815783876],
        [0, 2.0 * np.pi, 2, 0.4244131815783876],
        [0, -2.0 * np.pi, 2, 0.4244131815783876],
        [0, np.pi / 4, 2, 0.309207662451413],
        [0, -np.pi / 4, 2, 0.309207662451413],
        [0, np.pi / 4 + 2 * 2.0 * np.pi, 2, 0.309207662451413],
        [0, 7 * np.pi / 4.0, 0.5, 0.23096988312782166],
    ]

    @pytest.mark.parametrize("f,d,s,spread_expect", params__call__radians)
    def test__call__radians(self, f, d, s, spread_expect):
        spreading = CosineFullSpreading(s, degrees=False)
        assert spreading(f, d) == pytest.approx(spread_expect)

    def test_integrate_degrees(self):
        def integrate(spread_fun, a, b):
            f0 = 1
            return quad(lambda d: spread_fun(f0, d), a, b)[0]

        for s in (0, 1, 2, 10, 20):
            spreading = CosineFullSpreading(s, degrees=True)
            assert integrate(spreading, 0.0, 360.0) == pytest.approx(1)

    def test_integrate_radians(self):
        def integrate(spread_fun, a, b):
            f0 = 1
            return quad(lambda d: spread_fun(f0, d), a, b)[0]

        for s in (0, 1, 2, 10, 20):
            spreading = CosineFullSpreading(s, degrees=False)
            assert integrate(spreading, 0.0, 2.0 * np.pi) == pytest.approx(1)

    def test_independent_of_frequency(self):
        spreading = CosineFullSpreading(10, degrees=True)

        d0 = 45.0
        spread_out_list = [spreading(fi, d0) for fi in (0, 0.5, 1, 10)]

        assert len(np.unique(np.array(spread_out_list))) == 1

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([-26.4, 26.4])],
            [3, np.array([-37.6, 0.0, 37.6])],
            [4, np.array([-44.6, -12.5, 12.5, 44.6])],
            [5, np.array([-49.5, -20.5, 0.0, 20.5, 49.5])],
        ],
    )
    def test_discrete_directions_no_offset_full(self, n, dirs_expect):
        spreading = CosineFullSpreading(4, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineFullSpreading(4, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([-16.4, 36.4])],
            [3, np.array([-27.6, 10.0, 47.6])],
            [4, np.array([-34.6, -2.5, 22.5, 54.6])],
            [5, np.array([-39.5, -10.5, 10.0, 30.5, 59.5])],
        ],
    )
    def test_discrete_directions_offset_full(self, n, dirs_expect):
        spreading = CosineFullSpreading(4, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n, 10.0) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineFullSpreading(4, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n, np.radians(10)) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([-36.4, 16.4])],
            [3, np.array([-47.6, -10.0, 27.6])],
            [4, np.array([-54.6, -22.5, 2.5, 34.6])],
            [5, np.array([-59.5, -30.5, -10.0, 10.5, 39.5])],
        ],
    )
    def test_discrete_directions_neg_offset_full(self, n, dirs_expect):
        spreading = CosineFullSpreading(4, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n, -10.0) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineFullSpreading(4, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n, np.radians(-10)) + 1e-8)
            % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([332.6, 385.4])],
            [3, np.array([321.4, 359.0, 396.6])],
            [4, np.array([314.4, 346.5, 371.5, 403.6])],
            [5, np.array([309.5, 338.5, 359.0, 379.5, 408.5])],
        ],
    )
    def test_discrete_directions_large_offset_full(self, n, dirs_expect):
        spreading = CosineFullSpreading(4, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n, 359) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineFullSpreading(4, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n, np.radians(359)) + 1e-8)
            % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([-11.0, 31.0])],
            [3, np.array([-20.0, 10.0, 40.0])],
            [4, np.array([-25.6, 0.1, 19.9, 45.6])],
            [5, np.array([-29.5, -6.3, 10.0, 26.3, 49.5])],
        ],
    )
    def test_discrete_directions_other_s_full(self, n, dirs_expect):
        spreading = CosineFullSpreading(6.5, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n, 10) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineFullSpreading(6.5, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n, np.radians(10)) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )


class Test_CosineHalfSpreading:
    def test__init__(self):
        spreading = CosineHalfSpreading(123, degrees=True)
        assert isinstance(spreading, wr._core.BaseSpreading)
        assert spreading._s == 123
        assert spreading._degrees is True

    def test_integrate_degrees(self):
        def integrate(spread_fun, a, b):
            f0 = 1
            return quad(lambda d: spread_fun(f0, d), a, b)[0]

        for s in (0, 1, 2, 10, 20):
            spreading = CosineHalfSpreading(s, degrees=True)
            assert integrate(spreading, 0.0, 360.0) == pytest.approx(1)

    def test_integrate_radians(self):
        def integrate(spread_fun, a, b):
            f0 = 1
            return quad(lambda d: spread_fun(f0, d), a, b)[0]

        for s in (0, 1, 2, 10, 20):
            spreading = CosineHalfSpreading(s, degrees=False)
            assert integrate(spreading, 0.0, 2.0 * np.pi) == pytest.approx(1)

    def test_independent_of_frequency(self):
        spreading = CosineHalfSpreading(10, degrees=True)

        d0 = 45.0
        spread_out_list = [spreading(fi, d0) for fi in (0, 0.5, 1, 10)]

        assert len(np.unique(np.array(spread_out_list))) == 1

    params__call__degrees = [
        [0, 0, 0, 0.005555555555555556],
        [1, 0, 0, 0.005555555555555556],
        [0, 360, 0, 0.005555555555555556],
        [0, -360, 0, 0.005555555555555556],
        [0, 45, 0, 0.005555555555555556],
        [0, -45, 0, 0.005555555555555556],
        [0, 45 + 2 * 360, 0, 0.005555555555555556],
        [0, 0, 1, 0.011111111111111112],
        [1, 0, 1, 0.011111111111111112],
        [0, 360, 1, 0.011111111111111112],
        [0, -360, 1, 0.011111111111111112],
        [0, 45, 1, 0.005555555555555557],
        [0, -45, 1, 0.005555555555555557],
        [0, 45 + 2 * 360, 1, 0.005555555555555557],
        [0, 0, 2, 0.014814814814814815],
        [1, 0, 2, 0.014814814814814815],
        [0, 360, 2, 0.014814814814814815],
        [0, -360, 2, 0.014814814814814815],
        [0, 45, 2, 0.0037037037037037047],
        [0, -45, 2, 0.0037037037037037047],
        [0, 45 + 2 * 360, 2, 0.0037037037037037047],
        [0, 91, 0, 0.0],
        [0, -91, 0, 0.0],
        [0, 180, 0, 0.0],
        [0, -180, 0, 0.0],
        [0, 91, 1, 0.0],
        [0, -91, 1, 0.0],
        [0, 180, 1, 0.0],
        [0, -180, 1, 0.0],
        [0, 91, 10, 0.0],
        [0, -91, 10, 0.0],
        [0, 180, 10, 0.0],
        [0, -180, 10, 0.0],
    ]

    @pytest.mark.parametrize("f,d,s,spread_expect", params__call__degrees)
    def test__call__degrees(self, f, d, s, spread_expect):
        spreading = CosineHalfSpreading(s, degrees=True)
        assert spreading(f, d) == pytest.approx(spread_expect)

    params__call__radians = [
        [0, 0, 0, 0.3183098861837907],
        [1, 0, 0, 0.3183098861837907],
        [0, 2.0 * np.pi, 0, 0.3183098861837907],
        [0, -2.0 * np.pi, 0, 0.3183098861837907],
        [0, np.pi / 4, 0, 0.3183098861837907],
        [0, -np.pi / 4, 0, 0.3183098861837907],
        [0, (np.pi / 4) + 2 * 2.0 * np.pi, 0, 0.3183098861837907],
        [0, 0, 1, 0.6366197723675814],
        [1, 0, 1, 0.6366197723675814],
        [0, 2.0 * np.pi, 1, 0.6366197723675814],
        [0, -2.0 * np.pi, 1, 0.6366197723675814],
        [0, np.pi / 4, 1, 0.31830988618379075],
        [0, -np.pi / 4, 1, 0.31830988618379075],
        [0, np.pi / 4 + 2 * 2.0 * np.pi, 1, 0.31830988618379075],
        [0, 0, 2, 0.8488263631567752],
        [1, 0, 2, 0.8488263631567752],
        [0, 2.0 * np.pi, 2, 0.8488263631567752],
        [0, -2.0 * np.pi, 2, 0.8488263631567752],
        [0, np.pi / 4, 2, 0.21220659078919385],
        [0, -np.pi / 4, 2, 0.21220659078919385],
        [0, np.pi / 4 + 2 * 2.0 * np.pi, 2, 0.21220659078919385],
        [0, np.pi / 2 + 0.1, 0, 0.0],
        [0, -np.pi / 2 - 0.1, 0, 0.0],
        [0, np.pi, 0, 0.0],
        [0, -np.pi, 0, 0.0],
        [0, np.pi / 2 + 0.1, 1, 0.0],
        [0, -np.pi / 2 - 0.1, 1, 0.0],
        [0, np.pi, 1, 0.0],
        [0, -np.pi, 1, 0.0],
        [0, np.pi / 2 + 0.1, 10, 0.0],
        [0, -np.pi / 2 - 0.1, 10, 0.0],
        [0, np.pi, 10, 0.0],
        [0, -np.pi, 10, 0.0],
    ]

    @pytest.mark.parametrize("f,d,s,spread_expect", params__call__radians)
    def test__call__radians(self, f, d, s, spread_expect):
        spreading = CosineHalfSpreading(s, degrees=False)
        assert spreading(f, d) == pytest.approx(spread_expect)

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([-13.2, 13.2])],
            [3, np.array([-18.8, 0.0, 18.8])],
            [4, np.array([-22.3, -6.3, 6.3, 22.3])],
            [5, np.array([-24.8, -10.3, 0.0, 10.3, 24.8])],
        ],
    )
    def test_discrete_directions_no_offset(self, n, dirs_expect):
        spreading = CosineHalfSpreading(4, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineHalfSpreading(4, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([-3.2, 23.2])],
            [3, np.array([-8.8, 10.0, 28.8])],
            [4, np.array([-12.3, 3.7, 16.3, 32.3])],
            [5, np.array([-14.8, -0.3, 10.0, 20.3, 34.8])],
        ],
    )
    def test_discrete_directions_offset(self, n, dirs_expect):
        spreading = CosineHalfSpreading(4, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n, 10.0) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineHalfSpreading(4, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n, np.radians(10)) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([-23.2, 3.2])],
            [3, np.array([-28.8, -10.0, 8.8])],
            [4, np.array([-32.3, -16.3, -3.7, 12.3])],
            [5, np.array([-34.8, -20.3, -10.0, 0.3, 14.8])],
        ],
    )
    def test_discrete_directions_neg_offset(self, n, dirs_expect):
        spreading = CosineHalfSpreading(4, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n, -10.0) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineHalfSpreading(4, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n, np.radians(-10)) + 1e-8)
            % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([345.8, 372.2])],
            [3, np.array([340.2, 359, 377.8])],
            [4, np.array([336.7, 352.7, 365.3, 381.3])],
            [5, np.array([334.2, 348.7, 359.0, 369.3, 383.8])],
        ],
    )
    def test_discrete_directions_large_offset(self, n, dirs_expect):
        spreading = CosineHalfSpreading(4, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n, 359) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineHalfSpreading(4, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n, np.radians(359)) + 1e-8)
            % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

    @pytest.mark.parametrize(
        "n,dirs_expect",
        [
            [2, np.array([-0.5, 20.5])],
            [3, np.array([-5.0, 10.0, 25.0])],
            [4, np.array([-7.8, 5.0, 15.0, 27.8])],
            [5, np.array([-9.8, 1.8, 10.0, 18.2, 29.8])],
        ],
    )
    def test_discrete_directions_other_s(self, n, dirs_expect):
        spreading = CosineHalfSpreading(6.5, degrees=True)
        np.testing.assert_allclose(
            (spreading.discrete_directions(n, 10) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )

        spreading = CosineHalfSpreading(6.5, degrees=False)
        np.testing.assert_allclose(
            np.degrees(spreading.discrete_directions(n, np.radians(10)) + 1e-8) % 360.0,
            dirs_expect % 360.0,
            rtol=0.0,
            atol=0.1,
        )
