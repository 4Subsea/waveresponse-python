Rigid body transformation of RAOs
=================================



With `waveresponse` you can easily transform RAOs from one location on a rigid body,
*i*, to another, *j*, by calling the :meth:`~waveresponse.rigid_transform`.
Note that only the translational motion RAOs (surge, sway, heave) needs to be transformed;
the rotational degrees-of-freedom will be the same for all locations on a rigid body.

.. code-block:: python

    import numpy as np
    import waveresponse as wr


    t = np.array([10.0, 0.0, 0.0])
    surge_j, sway_j, heave_j = wr.rigid_transform_surge(
        t,
        surge_i,
        sway_j,
        heave_i,
        roll,
        pitch,
        yaw
    )
