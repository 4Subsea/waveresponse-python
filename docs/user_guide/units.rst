Units
=====


Convert RAO units
-----------------

.. When you do rigid body transformation of RAOs, it is required that the rotational
.. degree-of-freedom RAOs represents angles in *radians*. 

Rigid body transformation of RAOs require that the rotational degree-of-freedom
RAOs represent angles in *radians*. Then, it can be useful to be able to convert
an RAO from e.g. :math:`deg/m` units to :math:`rad/m` units. This is done by a scaling
of the RAO values with a factor :math:`\pi/180`.

With ``waveresponse`` you can convert an RAO object between different units simply
by scaling the RAO values with an appropriate factor:

.. code:: python

    import numpy as np


    # Convert RAO object from 'deg/m' to 'rad/m'
    deg2rad = np.pi / 180.0
    rao = deg2rad * rao

    # Convert RAO object from 'rad/m' to 'deg/m'
    rad2deg = 180.0 / np.pi
    rao = rad2deg * rao
