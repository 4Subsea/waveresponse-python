from pathlib import Path
import json

import numpy as np
import pytest
from scipy import integrate

from waveresponse import JONSWAP, ModifiedPiersonMoskowitz, BasePMSpectrum


TEST_PATH = Path(__file__).parent


class Test_ModifiedPiersonMoskowitz:
    def test__init___hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = ModifiedPiersonMoskowitz(freq, freq_hz=True)

        assert isinstance(spectrum, BasePMSpectrum)
        assert spectrum._freq_hz is True
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq)

    def test__init___rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = ModifiedPiersonMoskowitz(freq, freq_hz=False)

        assert isinstance(spectrum, BasePMSpectrum)
        assert spectrum._freq_hz is False
        np.testing.assert_array_almost_equal(spectrum._freq, freq)

    def test_A(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = ModifiedPiersonMoskowitz(freq)

        hs = 3.5
        tp = 10.0
        A_out = spectrum._A(hs, tp)

        omega_p = 2.0 * np.pi / tp
        A_expect = (5.0 / 16.0) * hs**2.0 * omega_p**4.0

        assert A_out == pytest.approx(A_expect)

    def test_B(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = ModifiedPiersonMoskowitz(freq)

        hs = 3.5
        tp = 10.0
        B_out = spectrum._B(hs, tp)

        omega_p = 2.0 * np.pi / tp
        B_expect = (5.0 / 4.0) * omega_p**4

        assert B_out == pytest.approx(B_expect)

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


class Test_JONSWAP:
    def test__init___hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq, freq_hz=True, gamma=3, sigma_a=0.1, sigma_b=0.2)

        assert isinstance(spectrum, BasePMSpectrum)
        assert isinstance(spectrum, ModifiedPiersonMoskowitz)
        assert spectrum._freq_hz is True
        assert spectrum._gamma == 3
        assert spectrum._sigma_a == 0.1
        assert spectrum._sigma_b == 0.2
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq)

    def test__init___rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq, freq_hz=False, gamma=3, sigma_a=0.1, sigma_b=0.2)

        assert isinstance(spectrum, BasePMSpectrum)
        assert isinstance(spectrum, ModifiedPiersonMoskowitz)
        assert spectrum._freq_hz is False
        assert spectrum._gamma == 3
        assert spectrum._sigma_a == 0.1
        assert spectrum._sigma_b == 0.2
        np.testing.assert_array_almost_equal(spectrum._freq, freq)

    def test_A(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq)

        hs = 3.5
        tp = 10.0
        A_out = spectrum._A(hs, tp)

        omega_p = 2.0 * np.pi / tp
        A_expect = (5.0 / 16.0) * hs**2.0 * omega_p**4.0

        assert A_out == pytest.approx(A_expect)

    def test_B(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq)

        hs = 3.5
        tp = 10.0
        B_out = spectrum._B(hs, tp)

        omega_p = 2.0 * np.pi / tp
        B_expect = (5.0 / 4.0) * omega_p**4

        assert B_out == pytest.approx(B_expect)

    def test__call__var(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = JONSWAP(freq)

        freq_rad, ps_rad = spectrum(3.5, 10.0, freq_hz=False)
        var_rad = integrate.trapz(ps_rad, freq_rad)

        freq_hz, ps_hz = spectrum(3.5, 10.0, freq_hz=True)
        var_hz = integrate.trapz(ps_hz, freq_hz)

        assert var_rad == pytest.approx(var_hz)

    def test__call__hz(self):
        hs = 3.5
        tp = 7.0
        gamma = 2
        sigma_a = 0.07
        sigma_b = 0.09
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq, freq_hz=True, gamma=gamma, sigma_a=sigma_a, sigma_b=sigma_b)
        freq_out, spectrum_out = spectrum(hs, tp, freq_hz=True)

        w_p = 2.0 * np.pi / tp
        A = (5.0 / 16.0) * hs**2 * w_p**4
        B = (5.0 / 4.0) * w_p**4
        w = 2.0 * np.pi * freq

        arg = w <= w_p
        sigma = np.empty_like(w)
        sigma[arg] = sigma_a
        sigma[~arg] = sigma_b

        alpha = 1.0 - 0.287 * np.log(gamma)
        b = np.exp(-0.5 * ((w - w_p) / (sigma * w_p)) ** 2)

        spectrum_expect = alpha * A / w**5 * np.exp(-B / w**4) * gamma ** b
        spectrum_expect *= 2.0 * np.pi

        np.testing.assert_array_almost_equal(freq_out, freq)
        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)

    def test__call__hz2(self):
        hs = 5.0
        tp = 10.0
        gamma = 3
        sigma_a = 0.05
        sigma_b = 0.08
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq, freq_hz=True, gamma=gamma, sigma_a=sigma_a, sigma_b=sigma_b)
        freq_out, spectrum_out = spectrum(hs, tp, freq_hz=True)

        w_p = 2.0 * np.pi / tp
        A = (5.0 / 16.0) * hs**2 * w_p**4
        B = (5.0 / 4.0) * w_p**4
        w = 2.0 * np.pi * freq

        arg = w <= w_p
        sigma = np.empty_like(w)
        sigma[arg] = sigma_a
        sigma[~arg] = sigma_b

        alpha = 1.0 - 0.287 * np.log(gamma)
        b = np.exp(-0.5 * ((w - w_p) / (sigma * w_p)) ** 2)

        spectrum_expect = alpha * A / w**5 * np.exp(-B / w**4) * gamma ** b
        spectrum_expect *= 2.0 * np.pi

        np.testing.assert_array_almost_equal(freq_out, freq)
        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)

    def test__call__rads(self):
        hs = 3.5
        tp = 7.0
        gamma = 2
        sigma_a = 0.07
        sigma_b = 0.09
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq, freq_hz=True, gamma=gamma, sigma_a=sigma_a, sigma_b=sigma_b)
        freq_out, spectrum_out = spectrum(hs, tp, freq_hz=False)

        w_p = 2.0 * np.pi / tp
        A = (5.0 / 16.0) * hs**2 * w_p**4
        B = (5.0 / 4.0) * w_p**4
        w = 2.0 * np.pi * freq

        arg = w <= w_p
        sigma = np.empty_like(w)
        sigma[arg] = sigma_a
        sigma[~arg] = sigma_b

        alpha = 1.0 - 0.287 * np.log(gamma)
        b = np.exp(-0.5 * ((w - w_p) / (sigma * w_p)) ** 2)

        spectrum_expect = alpha * A / w**5 * np.exp(-B / w**4) * gamma ** b

        np.testing.assert_array_almost_equal(freq_out, 2.0 * np.pi * freq)
        np.testing.assert_array_almost_equal(spectrum_out, spectrum_expect)

    def test__sigma(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = JONSWAP(freq, gamma=1)

        tp = 10.0
        omega_p = (2.0 * np.pi) / tp

        sigma = spectrum._sigma(omega_p)

        arg = freq <= omega_p
        sigma_1 = 0.07 * np.ones(sum(arg))
        sigma_2 = 0.09 * np.ones(len(arg) - sum(arg))

        np.testing.assert_array_equal(sigma[arg], sigma_1)
        np.testing.assert_array_equal(sigma[~arg], sigma_2)

    def test_reference_case(self):
        """
        Integration test. Compares to wave spectrum generated by OrcaFlex.
        """

        with open(TEST_PATH / "testdata" / "hs0350_tp0550_gamma3_jonswap.json") as f:
            data = json.load(f)

        freq_expected = np.array(data["freq"])
        ps_expected = np.array(data["jonswap"])

        spectrum = JONSWAP(freq_expected, gamma=3, freq_hz=False)
        freq_out, ps_out = spectrum(3.5, 5.5)

        np.testing.assert_array_almost_equal(freq_out, freq_expected)
        np.testing.assert_array_almost_equal(ps_out, ps_expected, decimal=2)
