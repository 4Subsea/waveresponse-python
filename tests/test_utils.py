import numpy as np
import pytest

from waveresponse._utils import _robust_modulus, complex_to_polar, polar_to_complex


class Test__robust_modulus:
    params_robust_modulus = [
        (2.5, 2.0, 0.5),
        (2.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
        (4.0, 2.0, 0.0),
        (4.5, 2.0, 0.5),
        (360.5, 360.0, 0.5),
        (360.0, 360.0, 0.0),
        (0.0, 360.0, 0.0),
        (2.0 * 360.0, 360.0, 0.0),
        (2.0 * 360.0 + 0.5, 360.0, 0.5),
        (2.0 * np.pi + 0.5, 2.0 * np.pi, 0.5),
        (2.0 * np.pi, 2.0 * np.pi, 0.0),
        (0.0, 2.0 * np.pi, 0.0),
        (2.0 * 2.0 * np.pi, 2.0 * np.pi, 0.0),
        (2.0 * 2.0 * np.pi + 0.5, 2.0 * np.pi, 0.5),
        (350.0 + 10.0, 360.0, 0.0),
    ]

    @pytest.mark.parametrize("x,period,out_expect", params_robust_modulus)
    def test__robust_modulus(self, x, period, out_expect):
        x_mod = _robust_modulus(x, period)
        assert x_mod == pytest.approx(out_expect)
        assert x_mod != period

    def test_array_rad(self):
        x = np.array([0.0, 0.5, 2.0 * np.pi, 2.0 * np.pi + 0.5])
        x_mod = _robust_modulus(x, 2.0 * np.pi)
        expect = np.array([0.0, 0.5, 0.0, 0.5])

        np.testing.assert_almost_equal(x_mod, expect)

    def test_array_deg(self):
        x = np.array([0.0, 0.5, 360.0, 360.0 + 0.5])
        x_mod = _robust_modulus(x, 360.0)
        expect = np.array([0.0, 0.5, 0.0, 0.5])

        np.testing.assert_almost_equal(x_mod, expect)


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
