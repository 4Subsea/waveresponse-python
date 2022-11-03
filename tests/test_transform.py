import numpy as np
import pytest

from waveresponse import (
    RAO,
    Grid,
    rigid_transform,
    rigid_transform_heave,
    rigid_transform_surge,
    rigid_transform_sway,
)


class Test_rigid_transform:
    def test_rigid_transform(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_surge = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_sway = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_heave = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [4.0 + 0.0j, 0.0 + 4.0j],
                [4.0 + 4.0j, 0.0 + 0.0j],
                [0.0 + 4.0j, 4.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [5.0 + 0.0j, 0.0 + 5.0j],
                [5.0 + 5.0j, 0.0 + 0.0j],
                [0.0 + 5.0j, 5.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [6.0 + 0.0j, 0.0 + 6.0j],
                [6.0 + 6.0j, 0.0 + 0.0j],
                [0.0 + 6.0j, 6.0 + 0.0j],
            ]
        )

        surge = RAO(freq, dirs, vals_surge, degrees=True)
        sway = RAO(freq, dirs, vals_sway, degrees=True)
        heave = RAO(freq, dirs, vals_heave, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        surge_out, sway_out, heave_out = rigid_transform(
            t, surge, sway, heave, roll, pitch, yaw
        )

        vals_surge_expect = vals_surge - 50.0 * vals_yaw + 60.0 * vals_pitch
        vals_sway_expect = vals_sway + 40.0 * vals_yaw - 60.0 * vals_roll
        vals_heave_expect = vals_heave - 40.0 * vals_pitch + 50.0 * vals_roll

        assert isinstance(surge_out, RAO)
        np.testing.assert_array_almost_equal(surge_out._freq, surge._freq)
        np.testing.assert_array_almost_equal(surge_out._dirs, surge._dirs)
        np.testing.assert_array_almost_equal(surge_out._vals, vals_surge_expect)
        assert surge_out._clockwise == surge._clockwise
        assert surge_out._waves_coming_from == surge._waves_coming_from

        assert isinstance(sway_out, RAO)
        np.testing.assert_array_almost_equal(sway_out._freq, sway._freq)
        np.testing.assert_array_almost_equal(sway_out._dirs, sway._dirs)
        np.testing.assert_array_almost_equal(sway_out._vals, vals_sway_expect)
        assert surge_out._clockwise == sway._clockwise
        assert surge_out._waves_coming_from == sway._waves_coming_from

        assert isinstance(heave_out, RAO)
        np.testing.assert_array_almost_equal(heave_out._freq, heave._freq)
        np.testing.assert_array_almost_equal(heave_out._dirs, heave._dirs)
        np.testing.assert_array_almost_equal(heave_out._vals, vals_heave_expect)
        assert heave_out._clockwise == heave._clockwise
        assert heave_out._waves_coming_from == heave._waves_coming_from

    def test_rigid_transform_raises_type(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_surge = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_sway = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_heave = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [4.0 + 0.0j, 0.0 + 4.0j],
                [4.0 + 4.0j, 0.0 + 0.0j],
                [0.0 + 4.0j, 4.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [5.0 + 0.0j, 0.0 + 5.0j],
                [5.0 + 5.0j, 0.0 + 0.0j],
                [0.0 + 5.0j, 5.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [6.0 + 0.0j, 0.0 + 6.0j],
                [6.0 + 6.0j, 0.0 + 0.0j],
                [0.0 + 6.0j, 6.0 + 0.0j],
            ]
        )

        surge = Grid(freq, dirs, vals_surge, degrees=True)
        sway = RAO(freq, dirs, vals_sway, degrees=True)
        heave = RAO(freq, dirs, vals_heave, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        with pytest.raises(ValueError):
            rigid_transform(t, surge, sway, heave, roll, pitch, yaw)

    def test_rigid_transform_raises_dirs(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_surge = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_sway = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_heave = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [4.0 + 0.0j, 0.0 + 4.0j],
                [4.0 + 4.0j, 0.0 + 0.0j],
                [0.0 + 4.0j, 4.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [5.0 + 0.0j, 0.0 + 5.0j],
                [5.0 + 5.0j, 0.0 + 0.0j],
                [0.0 + 5.0j, 5.0 + 0.0j],
            ]
        )

        surge = RAO(freq, dirs, vals_surge, degrees=True)
        sway = RAO(freq, dirs, vals_sway, degrees=True)
        heave = RAO(freq, dirs, vals_heave, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)

        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0, 359.0])
        vals_yaw = np.array(
            [
                [6.0 + 0.0j, 0.0 + 6.0j, 1.0 + 0.0j],
                [6.0 + 6.0j, 0.0 + 0.0j, 0.0 + 1.0j],
                [0.0 + 6.0j, 6.0 + 0.0j, 1.0 + 1.0j],
            ]
        )
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        with pytest.raises(ValueError):
            rigid_transform(t, surge, sway, heave, roll, pitch, yaw)

    def test_rigid_transform_raises_freq(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_surge = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_sway = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_heave = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [4.0 + 0.0j, 0.0 + 4.0j],
                [4.0 + 4.0j, 0.0 + 0.0j],
                [0.0 + 4.0j, 4.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [5.0 + 0.0j, 0.0 + 5.0j],
                [5.0 + 5.0j, 0.0 + 0.0j],
                [0.0 + 5.0j, 5.0 + 0.0j],
            ]
        )

        surge = RAO(freq, dirs, vals_surge, degrees=True)
        sway = RAO(freq, dirs, vals_sway, degrees=True)
        heave = RAO(freq, dirs, vals_heave, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)

        freq = np.array([0.0, 0.5, 1.0, 2.0])
        dirs = np.array([0.0, 180.0])
        vals_yaw = np.array(
            [
                [6.0 + 0.0j, 0.0 + 6.0j],
                [6.0 + 6.0j, 0.0 + 0.0j],
                [0.0 + 6.0j, 6.0 + 0.0j],
                [0.0 + 6.0j, 6.0 + 0.0j],
            ]
        )
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        with pytest.raises(ValueError):
            rigid_transform(t, surge, sway, heave, roll, pitch, yaw)

    def test_rigid_transform_raises_convention(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_surge = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_sway = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_heave = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [4.0 + 0.0j, 0.0 + 4.0j],
                [4.0 + 4.0j, 0.0 + 0.0j],
                [0.0 + 4.0j, 4.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [5.0 + 0.0j, 0.0 + 5.0j],
                [5.0 + 5.0j, 0.0 + 0.0j],
                [0.0 + 5.0j, 5.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [6.0 + 0.0j, 0.0 + 6.0j],
                [6.0 + 6.0j, 0.0 + 0.0j],
                [0.0 + 6.0j, 6.0 + 0.0j],
            ]
        )

        surge = RAO(freq, dirs, vals_surge, degrees=True)
        sway = RAO(freq, dirs, vals_sway, degrees=True)
        heave = RAO(freq, dirs, vals_heave, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        surge.set_wave_convention(clockwise=True, waves_coming_from=True)
        pitch.set_wave_convention(clockwise=False, waves_coming_from=False)

        t = np.array([40, 50, 60])
        with pytest.raises(ValueError):
            rigid_transform(t, surge, sway, heave, roll, pitch, yaw)


class Test_rigid_transform_surge:
    def test_rigid_transform_surge(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_surge = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        surge = RAO(freq, dirs, vals_surge, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        surge_out = rigid_transform_surge(t, surge, pitch, yaw)

        vals_expect = vals_surge - 50.0 * vals_yaw + 60.0 * vals_pitch

        assert isinstance(surge_out, RAO)
        np.testing.assert_array_almost_equal(surge_out._freq, surge._freq)
        np.testing.assert_array_almost_equal(surge_out._dirs, surge._dirs)
        np.testing.assert_array_almost_equal(surge_out._vals, vals_expect)
        assert surge_out._clockwise == surge._clockwise
        assert surge_out._waves_coming_from == surge._waves_coming_from

    def test_rigid_transform_surge_rot_degrees(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_surge = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        surge = RAO(freq, dirs, vals_surge, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        surge_out = rigid_transform_surge(t, surge, pitch, yaw, rot_degrees=True)

        vals_expect = (
            vals_surge
            - 50.0 * (np.pi / 180.0) * vals_yaw
            + 60.0 * (np.pi / 180.0) * vals_pitch
        )

        assert isinstance(surge_out, RAO)
        np.testing.assert_array_almost_equal(surge_out._freq, surge._freq)
        np.testing.assert_array_almost_equal(surge_out._dirs, surge._dirs)
        np.testing.assert_array_almost_equal(surge_out._vals, vals_expect)
        assert surge_out._clockwise == surge._clockwise
        assert surge_out._waves_coming_from == surge._waves_coming_from

    def test_rigid_transform_surge_rot_radians(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_surge = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        surge = RAO(freq, dirs, vals_surge, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        surge_out = rigid_transform_surge(t, surge, pitch, yaw, rot_degrees=False)

        vals_expect = vals_surge - 50.0 * vals_yaw + 60.0 * vals_pitch

        assert isinstance(surge_out, RAO)
        np.testing.assert_array_almost_equal(surge_out._freq, surge._freq)
        np.testing.assert_array_almost_equal(surge_out._dirs, surge._dirs)
        np.testing.assert_array_almost_equal(surge_out._vals, vals_expect)
        assert surge_out._clockwise == surge._clockwise
        assert surge_out._waves_coming_from == surge._waves_coming_from


class Test_rigid_transform_sway:
    def test_rigid_transform_sway(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_sway = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        sway = RAO(freq, dirs, vals_sway, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        sway_out = rigid_transform_sway(t, sway, roll, yaw)

        vals_expect = vals_sway + 40.0 * vals_yaw - 60.0 * vals_roll

        assert isinstance(sway_out, RAO)
        np.testing.assert_array_almost_equal(sway_out._freq, sway._freq)
        np.testing.assert_array_almost_equal(sway_out._dirs, sway._dirs)
        np.testing.assert_array_almost_equal(sway_out._vals, vals_expect)
        assert sway_out._clockwise == sway._clockwise
        assert sway_out._waves_coming_from == sway._waves_coming_from

    def test_rot_degrees(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_sway = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        sway = RAO(freq, dirs, vals_sway, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        sway_out = rigid_transform_sway(t, sway, roll, yaw, rot_degrees=True)

        vals_expect = (
            vals_sway
            + 40.0 * (np.pi / 180.0) * vals_yaw
            - 60.0 * (np.pi / 180.0) * vals_roll
        )

        assert isinstance(sway_out, RAO)
        np.testing.assert_array_almost_equal(sway_out._freq, sway._freq)
        np.testing.assert_array_almost_equal(sway_out._dirs, sway._dirs)
        np.testing.assert_array_almost_equal(sway_out._vals, vals_expect)
        assert sway_out._clockwise == sway._clockwise
        assert sway_out._waves_coming_from == sway._waves_coming_from

    def test_rot_radians(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_sway = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_yaw = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        sway = RAO(freq, dirs, vals_sway, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        yaw = RAO(freq, dirs, vals_yaw, degrees=True)

        t = np.array([40, 50, 60])
        sway_out = rigid_transform_sway(t, sway, roll, yaw, rot_degrees=False)

        vals_expect = vals_sway + 40.0 * vals_yaw - 60.0 * vals_roll

        assert isinstance(sway_out, RAO)
        np.testing.assert_array_almost_equal(sway_out._freq, sway._freq)
        np.testing.assert_array_almost_equal(sway_out._dirs, sway._dirs)
        np.testing.assert_array_almost_equal(sway_out._vals, vals_expect)
        assert sway_out._clockwise == sway._clockwise
        assert sway_out._waves_coming_from == sway._waves_coming_from


class Test_rigid_transform_heave:
    def test_rigid_transform_heave(self):
        freq = np.array([0.0, 0.5, 1.0])
        dirs = np.array([0.0, 180.0])

        vals_heave = np.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],
                [1.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 1.0j, 1.0 + 0.0j],
            ]
        )

        vals_roll = np.array(
            [
                [2.0 + 0.0j, 0.0 + 2.0j],
                [2.0 + 2.0j, 0.0 + 0.0j],
                [0.0 + 2.0j, 2.0 + 0.0j],
            ]
        )

        vals_pitch = np.array(
            [
                [3.0 + 0.0j, 0.0 + 3.0j],
                [3.0 + 3.0j, 0.0 + 0.0j],
                [0.0 + 3.0j, 3.0 + 0.0j],
            ]
        )

        heave = RAO(freq, dirs, vals_heave, degrees=True)
        roll = RAO(freq, dirs, vals_roll, degrees=True)
        pitch = RAO(freq, dirs, vals_pitch, degrees=True)

        t = np.array([40, 50, 60])
        heave_out = rigid_transform_heave(t, heave, roll, pitch)

        vals_expect = vals_heave - 40.0 * vals_pitch + 50.0 * vals_roll

        assert isinstance(heave_out, RAO)
        np.testing.assert_array_almost_equal(heave_out._freq, heave._freq)
        np.testing.assert_array_almost_equal(heave_out._dirs, heave._dirs)
        np.testing.assert_array_almost_equal(heave_out._vals, vals_expect)
        assert heave_out._clockwise == heave._clockwise
        assert heave_out._waves_coming_from == heave._waves_coming_from
