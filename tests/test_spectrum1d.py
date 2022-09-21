import numpy as np
import pytest
from scipy import integrate

from waveresponse import JONSWAP, ModifiedPiersonMoskowitz


class Test_ModifiedPiersonMoskowitz:
    def test__init___hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = ModifiedPiersonMoskowitz(freq, freq_hz=True)

        assert spectrum._freq_hz is True
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq)

    def test__init___rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = ModifiedPiersonMoskowitz(freq, freq_hz=False)

        assert spectrum._freq_hz is False
        np.testing.assert_array_almost_equal(spectrum._freq, freq)

    def test__call__hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = ModifiedPiersonMoskowitz(freq, freq_hz=True)

        hs = 3.5
        tp = 7.0
        freq_out, spectrum_out = spectrum(hs, tp, freq_hz=True)

        w_p = 2.0 * np.pi / tp
        A = (5.0 / 16.0) * hs**2 * w_p**4
        B = (5.0 / 4.0) * w_p**4
        w = 2.0 * np.pi * freq
        spectrum_expect = A / w**5 * np.exp(-B / w**4)
        spectrum_expect *= 2.0 * np.pi

        np.testing.assert_array_almost_equal(freq_out, freq)
        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)

    def test__call__rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = ModifiedPiersonMoskowitz(freq, freq_hz=True)

        hs = 3.5
        tp = 7.0
        freq_out, spectrum_out = spectrum(hs, tp, freq_hz=False)

        w_p = 2.0 * np.pi / tp
        A = (5.0 / 16.0) * hs**2 * w_p**4
        B = (5.0 / 4.0) * w_p**4
        w = 2.0 * np.pi * freq
        spectrum_expect = A / w**5 * np.exp(-B / w**4)

        np.testing.assert_array_almost_equal(freq_out, 2.0 * np.pi * freq)
        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)

    def test__call__var(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = ModifiedPiersonMoskowitz(freq)

        freq_rad, ps_rad = spectrum(3.5, 10.0, freq_hz=False)
        var_rad = integrate.trapz(ps_rad, freq_rad)

        freq_hz, ps_hz = spectrum(3.5, 10.0, freq_hz=True)
        var_hz = integrate.trapz(ps_hz, freq_hz)

        assert var_rad == pytest.approx(var_hz)
