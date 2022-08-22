import numpy as np


def rigid_transform(surge, sway, heave, roll, pitch, yaw, t):
    """
    Rigid body transformation of response amplitude operators (RAOs).

    Parameters
    ----------
    surge : obj
        Surge RAO as a :class:`~waveresponse.RAO` object.
    sway : obj
        Sway RAO as a :class:`~waveresponse.RAO` object.
    heave : obj
        Heave RAO as a :class:`~waveresponse.RAO` object.
    roll : obj
        Roll RAO as a :class:`~waveresponse.RAO` object.
    pitch : obj
        Pitch RAO as a :class:`~waveresponse.RAO` object.
    yaw : obj
        Yaw RAO as a :class:`~waveresponse.RAO` object.
    t : array-like (3,)
        Translation vector given as (x, y, z).
    """
    t = np.asarray_chkfinite(t).reshape(-1, 1)

    if t.shape != (3, 1):
        raise ValueError("Translation vector should have length 3.")

    r = np.array([surge, sway, heave]).reshape(-1, 1)

    rot_matrix = np.array([
        [1.0, -yaw, pitch],
        [yaw, 1.0, -roll],
        [-pitch, roll, 1.0]
    ])

    surge_new, sway_new, heave_new = r + rot_matrix.dot(t)

    return surge_new, sway_new, heave_new, roll, pitch, yaw


def rigid_transform_surge(t, surge, pitch, yaw):
    """
    Rigid body transformation of surge response amplitude operator (RAO).

    Parameters
    ----------
    surge : obj
        Surge RAO as a :class:`~waveresponse.RAO` object.
    pitch : obj
        Pitch RAO as a :class:`~waveresponse.RAO` object.
    yaw : obj
        Yaw RAO as a :class:`~waveresponse.RAO` object.
    t : array-like
        Translation vector.
    """
    t = np.asarray_chkfinite(t)

    if len(t) != 3:
        raise ValueError("Translation vector should have length 3.")

    return surge - t[1] * yaw + t[2] * pitch


def rigid_transform_sway(t, sway, roll, yaw):
    """
    Rigid body transformation of sway response amplitude operator (RAO).

    Parameters
    ----------
    sway : obj
        Sway RAO as a :class:`~waveresponse.RAO` object.
    roll : obj
        Roll RAO as a :class:`~waveresponse.RAO` object.
    yaw : obj
        Yaw RAO as a :class:`~waveresponse.RAO` object.
    t : array-like
        Translation vector.
    """
    t = np.asarray_chkfinite(t)

    if len(t) != 3:
        raise ValueError("Translation vector should have length 3.")

    return sway + t[0] * yaw - t[2] * roll


def rigid_transform_heave(t, heave, roll, pitch):
    """
    Rigid body transformation of heave response amplitude operator (RAO).

    Parameters
    ----------
    heave : obj
        Heave RAO as a :class:`~waveresponse.RAO` object.
    roll : obj
        Roll RAO as a :class:`~waveresponse.RAO` object.
    pitch : obj
        Pitch RAO as a :class:`~waveresponse.RAO` object.
    t : array-like
        Translation vector.
    """
    t = np.asarray_chkfinite(t)

    if len(t) != 3:
        raise ValueError("Translation vector should have length 3.")

    return heave - t[0] * pitch + t[1] * roll
