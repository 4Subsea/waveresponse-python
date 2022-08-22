import numpy as np

from ._core import RAO


def rigid_transform(t, surge, sway, heave, roll, pitch, yaw):
    """
    Rigid body transformation of RAOs.

    Parameters
    ----------
    t : array-like
        Translation vector.
    surge : obj
        Surge RAO.
    sway : obj
        Sway RAO.
    heave : obj
        Heave RAO.
    roll : obj
        Roll RAO.
    pitch : obj
        Pitch RAO.
    yaw : obj
        Yaw RAO.
    """
    surge_new = rigid_transform_surge(t, surge, pitch, yaw)
    sway_new = rigid_transform_sway(t, sway, roll, yaw)
    heave_new = rigid_transform_heave(t, heave, roll, pitch)

    return surge_new, sway_new, heave_new, roll.copy(), pitch.copy(), yaw.copy()


def rigid_transform_surge(t, surge, pitch, yaw):
    """
    Rigid body transformation of surge RAO.

    Parameters
    ----------
    t : array-like
        Translation vector.
    surge : obj
        Surge RAO.
    pitch : obj
        Pitch RAO.
    yaw : obj
        Yaw RAO.
    """
    t = np.asarray_chkfinite(t)

    if len(t) != 3:
        raise ValueError("Translation vector should have length 3.")

    if not isinstance(surge, RAO):
        raise ValueError()

    surge_new = surge.copy()
    surge_new._check_similar(surge, pitch, yaw, exact_type=True)
    surge_new._vals = surge._vals - t[1] * yaw._vals + t[2] * pitch._vals

    return surge_new


def rigid_transform_sway(t, sway, roll, yaw):
    """
    Rigid body transformation of sway RAO.

    Parameters
    ----------
    t : array-like
        Translation vector.
    sway : obj
        Sway RAO.
    roll : obj
        Roll RAO.
    yaw : obj
        Yaw RAO.
    """
    t = np.asarray_chkfinite(t)

    if len(t) != 3:
        raise ValueError("Translation vector should have length 3.")

    if not isinstance(sway, RAO):
        raise ValueError()

    sway_new = sway.copy()
    sway_new._check_similar(sway, roll, yaw, exact_type=True)
    sway_new._vals = sway._vals + t[0] * yaw._vals - t[2] * roll._vals

    return sway_new


def rigid_transform_heave(t, heave, roll, pitch):
    """
    Rigid body transformation of heave RAO.

    Parameters
    ----------
    heave : obj
        Heave RAO.
    roll : obj
        Roll RAO.
    pitch : obj
        Pitch RAO.
    t : array-like
        Translation vector.
    """
    t = np.asarray_chkfinite(t)

    if len(t) != 3:
        raise ValueError("Translation vector should have length 3.")

    if not isinstance(heave, RAO):
        raise ValueError()

    heave_new = heave.copy()
    heave_new._check_similar(heave, roll, pitch, exact_type=True)
    heave_new._vals = heave._vals - t[0] * pitch._vals + t[1] * roll._vals

    return heave_new
