import numpy as np

from ._core import RAO


def rigid_transform(
    t: np.array, surge: RAO, sway: RAO, heave: RAO, roll: RAO, pitch: RAO, yaw: RAO
):
    """
    Rigid body transformation of (surge, sway and heave) RAOs.

    Transforms surge, sway and heave RAOs from one location to another by assuming
    rigid body motion. Note that the rotational degrees-of-freedom (i.e., roll,
    pitch and yaw) does not need transformation, since these are independent of
    location, and thus will be the same for all points on the rigid body.

    Parameters
    ----------
    t : array-like
        Translation vector, describing the translation 'from-old-to-new' location
        on the rigid body. Given as ``(x_new, y_new, z_new) - (x_old, y_old, z_old)``,
        where ``(x_new, y_new, z_new)`` are coordinates of the 'new' location, and
        ``(x_old, y_old, z_old)`` are coordinates of the 'old' location.
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

    Returns
    -------
    surge_new : obj
        Surge RAO (rigid body transformed).
    sway_new : obj
        Sway RAO (rigid body transformed).
    heave_new : obj
        Heave RAO (rigid body transformed).
    """
    surge_new = rigid_transform_surge(t, surge, pitch, yaw)
    sway_new = rigid_transform_sway(t, sway, roll, yaw)
    heave_new = rigid_transform_heave(t, heave, roll, pitch)

    return surge_new, sway_new, heave_new


def rigid_transform_surge(
    t: np.array, surge: RAO, pitch: RAO, yaw: RAO, rot_degrees: bool = False
) -> RAO:
    """
    Rigid body transformation of surge RAO.

    Transforms a surge RAO from one location to another on a rigid body.

    Parameters
    ----------
    t : array-like
        Translation vector, describing the translation 'from-old-to-new' location
        on the rigid body. Given as ``(x_new, y_new, z_new) - (x_old, y_old, z_old)``,
        where ``(x_new, y_new, z_new)`` are coordinates of the 'new' location, and
        ``(x_old, y_old, z_old)`` are coordinates of the 'old' location.
    surge : obj
        Surge RAO.
    pitch : obj
        Pitch RAO.
    yaw : obj
        Yaw RAO.
    rot_degrees : bool
        Weather the rotational degree-of-freedom RAOs (i.e., pitch and yaw) has
        amplitudes given in 'deg/m' units. If ``False``, 'rad/m' is assumed.

    Returns
    -------
    surge_new : obj
        Surge RAO (rigid body transformed).
    """
    t = np.asarray_chkfinite(t)

    try:
        _, ty, tz = t
    except ValueError:
        raise ValueError("Translation vector, `t`, should have length 3.")

    if not isinstance(surge, RAO):
        raise ValueError("RAO objects must be of type 'waveresponse.RAO'.")

    if rot_degrees:
        rot_scale = np.pi / 180.0
    else:
        rot_scale = 1.0

    return surge - ty * rot_scale * yaw + tz * rot_scale * pitch


def rigid_transform_sway(
    t: np.array, sway: RAO, roll: RAO, yaw: RAO, rot_degrees: bool = False
) -> RAO:
    """
    Rigid body transformation of sway RAO.

    Transforms a sway RAO from one location to another on a rigid body.

    Parameters
    ----------
    t : array-like
        Translation vector, describing the translation 'from-old-to-new' location
        on the rigid body. Given as ``(x_new, y_new, z_new) - (x_old, y_old, z_old)``,
        where ``(x_new, y_new, z_new)`` are coordinates of the 'new' location, and
        ``(x_old, y_old, z_old)`` are coordinates of the 'old' location.
    sway : obj
        Sway RAO.
    roll : obj
        Roll RAO.
    yaw : obj
        Yaw RAO.
    rot_degrees : bool
        Weather the rotational degree-of-freedom RAOs (i.e., roll and yaw) has
        amplitudes given in 'deg/m' units. If ``False``, 'rad/m' is assumed.

    Returns
    -------
    sway_new : obj
        Sway RAO (rigid body transformed).
    """
    t = np.asarray_chkfinite(t)

    try:
        tx, ty, tz = t
    except ValueError:
        raise ValueError("Translation vector, `t`, should have length 3.")

    if not isinstance(sway, RAO):
        raise ValueError("RAO objects must be of type 'waveresponse.RAO'.")

    if rot_degrees:
        rot_scale = np.pi / 180.0
    else:
        rot_scale = 1.0

    return sway + tx * rot_scale * yaw - tz * rot_scale * roll


def rigid_transform_heave(t: np.array, heave: RAO, roll: RAO, pitch: RAO) -> RAO:
    """
    Rigid body transformation of heave RAO.

    Transforms a heave RAO from one location to another on a rigid body.

    Parameters
    ----------
    t : array-like
        Translation vector, describing the translation 'from-old-to-new' location
        on the rigid body. Given as ``(x_new, y_new, z_new) - (x_old, y_old, z_old)``,
        where ``(x_new, y_new, z_new)`` are coordinates of the 'new' location, and
        ``(x_old, y_old, z_old)`` are coordinates of the 'old' location.
    heave : obj
        Heave RAO.
    roll : obj
        Roll RAO.
    pitch : obj
        Pitch RAO.

    Returns
    -------
    heave_new : obj
        Heave RAO (rigid body transformed).
    """
    t = np.asarray_chkfinite(t)

    try:
        tx, ty, tz = t
    except ValueError:
        raise ValueError("Translation vector, `t`, should have length 3.")

    if not isinstance(heave, RAO):
        raise ValueError("RAO objects must be of type 'waveresponse.RAO'.")

    return heave - tx * pitch + ty * roll
