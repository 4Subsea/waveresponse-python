import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from waveresponse import JONSWAP_ as JONSWAP
from waveresponse import BasePMSpectrum_ as BasePMSpectrum
from waveresponse import ModifiedPiersonMoskowitz_ as ModifiedPiersonMoskowitz
from waveresponse import OchiHubble_ as OchiHubble


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
        assert spectrum._sigma_a == 0.1
        assert spectrum._sigma_b == 0.2
        assert callable(spectrum._gamma)
        assert spectrum._gamma(1, 2) == 3
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq)

    def test__init___rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq, freq_hz=False, gamma=3, sigma_a=0.1, sigma_b=0.2)

        assert isinstance(spectrum, BasePMSpectrum)
        assert isinstance(spectrum, ModifiedPiersonMoskowitz)
        assert spectrum._freq_hz is False
        assert spectrum._sigma_a == 0.1
        assert spectrum._sigma_b == 0.2
        assert callable(spectrum._gamma)
        assert spectrum._gamma(1, 2) == 3
        np.testing.assert_array_almost_equal(spectrum._freq, freq)

    def test_gamma_fun(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq, gamma=lambda hs, tp: hs + tp + 2)
        assert spectrum._gamma(1, 2) == 5

    def test_gamma_float(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = JONSWAP(freq, gamma=1.2)
        assert spectrum._gamma(1, 2) == 1.2

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
        spectrum = JONSWAP(
            freq, freq_hz=True, gamma=gamma, sigma_a=sigma_a, sigma_b=sigma_b
        )
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

        spectrum_expect = alpha * A / w**5 * np.exp(-B / w**4) * gamma**b
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
        spectrum = JONSWAP(
            freq, freq_hz=True, gamma=gamma, sigma_a=sigma_a, sigma_b=sigma_b
        )
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

        spectrum_expect = alpha * A / w**5 * np.exp(-B / w**4) * gamma**b
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
        spectrum = JONSWAP(
            freq, freq_hz=True, gamma=gamma, sigma_a=sigma_a, sigma_b=sigma_b
        )
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

        spectrum_expect = alpha * A / w**5 * np.exp(-B / w**4) * gamma**b

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


class Test_OchiHubble:
    def test__init___hz(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = OchiHubble(freq, freq_hz=True, q=2.0)

        assert isinstance(spectrum, BasePMSpectrum)
        assert isinstance(spectrum, ModifiedPiersonMoskowitz)
        assert spectrum._freq_hz is True
        assert callable(spectrum._q)
        assert spectrum._q(1, 2) == 2.0
        np.testing.assert_array_almost_equal(spectrum._freq, 2.0 * np.pi * freq)

    def test__init___rads(self):
        freq = np.arange(0.01, 1, 0.01)
        spectrum = OchiHubble(freq, freq_hz=False, q=2.0)

        assert isinstance(spectrum, BasePMSpectrum)
        assert isinstance(spectrum, ModifiedPiersonMoskowitz)
        assert spectrum._freq_hz is False
        assert callable(spectrum._q)
        assert spectrum._q(1, 2) == 2.0
        np.testing.assert_array_almost_equal(spectrum._freq, freq)

    def test__call__var(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = OchiHubble(freq)

        freq_rad, ps_rad = spectrum(3.5, 10.0, freq_hz=False)
        var_rad = integrate.trapz(ps_rad, freq_rad)

        freq_hz, ps_hz = spectrum(3.5, 10.0, freq_hz=True)
        var_hz = integrate.trapz(ps_hz, freq_hz)

        assert var_rad == pytest.approx(var_hz)

    def test_A_q_1(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = OchiHubble(freq, q=1)

        hs = 3.5
        tp = 10.0
        omega_p = (2.0 * np.pi) / tp

        A = np.unique(spectrum._A(hs, tp))
        assert len(A) == 1
        assert 5 / 16 * hs**2 * omega_p**4 == A[0]

    def test_A_q_2(self):
        tp = 10.0
        omega_p = (2.0 * np.pi) / tp

        freq = np.array([omega_p])  # rad/s
        spectrum_q_20 = OchiHubble(freq, q=2.0)

        A_out = spectrum_q_20._A(3.5, tp)
        A_expected = 81.0 / 64.0 * 3.5**2 * omega_p**4

        assert A_expected == pytest.approx(A_out)

    def test_B_q_1(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = OchiHubble(freq, q=1)

        hs = 3.5
        tp = 10.0
        omega_p = (2.0 * np.pi) / tp

        B = np.unique(spectrum._B(hs, tp))
        assert len(B) == 1
        assert 1.25 * omega_p**4 == B[0]

    def test_B_q_2(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = OchiHubble(freq, q=2)

        hs = 3.5
        tp = 10.0
        omega_p = (2.0 * np.pi) / tp

        B = np.unique(spectrum._B(hs, tp))
        assert len(B) == 1
        assert 2.25 * omega_p**4 == B[0]

    def test_q_other_than_1(self):
        hs = 3.5
        tp = 10.0

        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        _, spectrum_ref = OchiHubble(freq, q=1)(hs, tp)
        _, spectrum_gamma_05 = OchiHubble(freq, q=0.5)(hs, tp)
        _, spectrum_gamma_20 = OchiHubble(freq, q=2.0)(hs, tp)

        assert max(spectrum_ref) > max(spectrum_gamma_05)
        assert max(spectrum_ref) < max(spectrum_gamma_20)

        assert spectrum_ref[-1] < spectrum_gamma_05[-1]
        assert spectrum_ref[-1] > spectrum_gamma_20[-1]

    def test_q_scalar(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = OchiHubble(freq, q=1)

        assert 1 == spectrum._q()

    def test_q_fun(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = OchiHubble(freq, q=lambda x: x + 2)

        assert 3 == spectrum._q(1)

    def test__call__freq_hz(self):
        freq = np.arange(0.05, 2.0, 0.05) * 2.0 * np.pi  # rad/s
        spectrum = OchiHubble(freq, freq_hz=False, q=1.0)

        hs = 3.5
        tp = 10.0
        spectrum_internal = spectrum._spectrum(freq, hs, tp)

        freq_out, spectrum_out = spectrum(3.5, 10.0, freq_hz=True)

        np.testing.assert_array_equal(freq_out, freq / (2.0 * np.pi))
        np.testing.assert_array_equal(spectrum_out, spectrum_internal * (2.0 * np.pi))

    def test_reference_case(self):
        """
        Integration test. Compares to wave spectrum generated by OrcaFlex.
        """

        data = pd.read_json(
            TEST_PATH / "testdata" / "hs0350_tp0550_q3_ochi_hubble.json"
        )
        freq_expected = data["freq"].values[1:]
        ps_expected = data["power_spectrum"].values[1:]

        spectrum = OchiHubble(freq_expected, freq_hz=True, q=3)
        freq_return, ps_return = spectrum(3.5, 5.5)

        np.testing.assert_array_almost_equal(freq_return, freq_expected)
        np.testing.assert_array_almost_equal(ps_return, ps_expected, decimal=2)
