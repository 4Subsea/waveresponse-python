Rigid body transformation of RAOs
=================================

Rigid body transformation of surge, sway and heave RAOs is governed by the following
equations:

.. math::

    H_{x_j}(\omega) = H_{x_i}(\omega) - y_{ij}H_{\gamma}(\omega) + z_{ij} H_{\theta}(\omega)

.. math::

    H_{y_j}(\omega) = H_{y_i}(\omega) + x_{ij}H_{\gamma}(\omega) - z_{ij}H_{\alpha}(\omega)

.. math::
    H_{z_j}(\omega) = H_{z_i}(\omega) - x_{ij}H_{\theta}(\omega) + y_{ij}H_{\alpha}(\omega)

where :math:`x_{ij}`, :math:`y_{ij}` and :math:`z_{ij}` describes the coordinates of the 'new' location,
*j*, relative to the 'old' location, *i*. :math:`H_x(\omega)` is the surge RAO,
:math:`H_y(\omega)` is the sway RAO, :math:`H_z(\omega)` is the heave RAO,
:math:`H_{\alpha}(\omega)` is the roll RAO, :math:`H_{\theta}(\omega)` is the pitch RAO,
and :math:`H_{\gamma}(\omega)` is the yaw RAO.

.. note::

    It is only the translational degrees-of-freedom (i.e., surge, sway and heave)
    that need to be transformed in order to obtain RAOs for a different location
    on a rigid body. The rotational motions (i.e., roll, pitch and yaw) are independent
    of location, and will be the same for all points on a rigid body.

With ``waveresponse`` you can easily transform RAOs from one location (*i*) to another (*j*)
on a rigid body by calling the :meth:`~waveresponse.rigid_transform` method. You
must then provide a translation vector, ``t``, that gives the coordinates of point
*j* relative to point *i*.

.. code-block:: python

    import numpy as np
    import waveresponse as wr


    t = np.array([10.0, 0.0, 0.0])   # (x, y, z)

    # Rigid transform surge, sway and heave
    surge_j, sway_j, heave_j = wr.rigid_transform(t, surge_i, sway_i, heave_i, roll, pitch, yaw)

    # Rigid transform heave only
    heave_j = wr.rigid_transform_heave(t, heave_i, roll, pitch)
