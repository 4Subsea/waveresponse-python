import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

import waveresponse as wr

TEST_PATH = Path(__file__).parent


class Test_BasePMSpectrum:
    def test__init___hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.BasePMSpectrum(freq, freq_hz=True)

        assert isinstance(spectrum, wr.BaseWave1d)
        assert spectrum._freq_hz is True
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq)

    def test__init___rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.BasePMSpectrum(freq, freq_hz=False)

        assert isinstance(spectrum, wr.BaseWave1d)
        assert spectrum._freq_hz is False
        np.testing.assert_array_almost_equal(spectrum._freq, freq)

    def test_spectrum(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.BasePMSpectrum(freq, freq_hz=False)

        omega = freq
        hs = 3.5
        tp = 10.0
        w_p = 2.0 * np.pi / tp
        A = 5 / 16 * hs ** 2 * w_p ** 4
        B = 5 / 4 * w_p ** 4
        spectrum_out = spectrum._spectrum(omega, A, B)

        spectrum_expect = A / omega**5.0 * np.exp(-B / omega**4)

        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)

    def test__call__hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.BasePMSpectrum(freq, freq_hz=False)

        omega = freq
        hs = 3.5
        tp = 10.0
        w_p = 2.0 * np.pi / tp
        A = 5 / 16 * hs ** 2 * w_p ** 4
        B = 5 / 4 * w_p ** 4
        freq_out, spectrum_out = spectrum(A, B, freq_hz=True)

        freq_expect = freq / (2.0 * np.pi)
        spectrum_expect = A / omega**5.0 * np.exp(-B / omega**4)
        spectrum_expect *= 2.0 * np.pi

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)

    def test__call__rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.BasePMSpectrum(freq, freq_hz=False)

        omega = freq
        hs = 3.5
        tp = 10.0
        w_p = 2.0 * np.pi / tp
        A = 5 / 16 * hs ** 2 * w_p ** 4
        B = 5 / 4 * w_p ** 4
        freq_out, spectrum_out = spectrum(A, B, freq_hz=False)

        freq_expect = freq
        spectrum_expect = A / omega**5.0 * np.exp(-B / omega**4)

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)

    def test__call__default(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.BasePMSpectrum(freq, freq_hz=False)

        omega = freq
        hs = 3.5
        tp = 10.0
        w_p = 2.0 * np.pi / tp
        A = 5 / 16 * hs ** 2 * w_p ** 4
        B = 5 / 4 * w_p ** 4
        freq_out, spectrum_out = spectrum(A, B, freq_hz=None)

        freq_expect = freq
        spectrum_expect = A / omega**5.0 * np.exp(-B / omega**4)

        np.testing.assert_array_almost_equal(freq_out, freq_expect)
        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)


class Test_ModifiedPiersonMoskowitz:
    def test__init___hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.ModifiedPiersonMoskowitz(freq, freq_hz=True)

        assert isinstance(spectrum, wr.BasePMSpectrum)
        assert spectrum._freq_hz is True
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq)

    def test__init___rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.ModifiedPiersonMoskowitz(freq, freq_hz=False)

        assert isinstance(spectrum, wr.BasePMSpectrum)
        assert spectrum._freq_hz is False
        np.testing.assert_array_almost_equal(spectrum._freq, freq)

    def test_A(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.ModifiedPiersonMoskowitz(freq)

        hs = 3.5
        tp = 10.0
        A_out = spectrum._A(hs, tp)

        omega_p = 2.0 * np.pi / tp
        A_expect = (5.0 / 16.0) * hs**2.0 * omega_p**4.0

        assert A_out == pytest.approx(A_expect)

    def test_B(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.ModifiedPiersonMoskowitz(freq)

        hs = 3.5
        tp = 10.0
        B_out = spectrum._B(hs, tp)

        omega_p = 2.0 * np.pi / tp
        B_expect = (5.0 / 4.0) * omega_p**4

        assert B_out == pytest.approx(B_expect)

    def test__call__hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = wr.ModifiedPiersonMoskowitz(freq, freq_hz=True)

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
        spectrum = wr.ModifiedPiersonMoskowitz(freq, freq_hz=True)

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
        spectrum = wr.ModifiedPiersonMoskowitz(freq)

        freq_rad, ps_rad = spectrum(3.5, 10.0, freq_hz=False)
        var_rad = integrate.trapz(ps_rad, freq_rad)

        freq_hz, ps_hz = spectrum(3.5, 10.0, freq_hz=True)
        var_hz = integrate.trapz(ps_hz, freq_hz)

        assert var_rad == pytest.approx(var_hz)
