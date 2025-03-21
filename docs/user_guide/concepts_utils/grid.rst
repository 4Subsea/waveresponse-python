Grid
====
The :class:`~waveresponse.Grid` class provides an interface for handling values
that should be interpreted on a two-dimentional frequency/(wave)direction grid.
:class:`~waveresponse.Grid` provides some useful functionality on its own, but the
main intension of this class is to serve as a base class. :class:`~waveresponse.RAO`,
:class:`~waveresponse.DirectionalSpectrum` and :class:`~waveresponse.WaveSpectrum`
are all examples of classes that extend :class:`~waveresponse.Grid`'s functionality.

The :class:`~waveresponse.Grid` class is initialized with a frequency list (1-D array),
a direction list (1-D array) and corresponding grid values (2-D array).

.. code-block:: python

    import numpy as np
    import waveresponse as wr


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, 10, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    grid = wr.Grid(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
    )

To properly interpret the (wave) directions and values that are associated with
a grid, we need information about the assumed 'wave convention'. Two boolean
parameters are needed:

*clockwise*
    Describes the direction of positive rotation. ``clockwise=True`` means that the
    directions follow the right-hand rule with an axis pointing downwards.
*waves_coming_from*
    Describes the direction in which the waves are propagating.

These parameters are set during initialization of the grid object:

.. code-block:: python

    grid = wr.Grid(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=True,
    )

.. tip::
    The grid values can be visualized e.g. using :mod:`matplotlib` and a polar plot:

    .. code:: python

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np


        f = np.linspace(0., 0.5, 50)   # Hz
        d = np.linspace(0., 2.0 * np.pi - 1e-8, 50)   # rad
        v = grid.interpolate(f, d, freq_hz=True, degrees=False)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        ax.contourf(d, f, v, levels=7, cmap=cm.jet)
        plt.show()

The grid can be converted to a different wave convention anytime by calling the
:meth:`~waveresponse.Grid.set_wave_convention` method with the desired convention flags.

.. code-block:: python

    grid.set_wave_convention(clockwise=False, waves_coming_from=True)

The frequency/direction coordinates and values of the :class:`~waveresponse.Grid`
instance can be retrieved by calling the :meth:`~waveresponse.Grid.grid` method.
You must then specify which coordinate units to return by setting the ``freq_hz``
and ``degrees`` flags.

.. code-block:: python

    freq, dirs, vals = grid.grid(freq_hz=True, degrees=True)

Interpolation of the grid values is provided by the :meth:`~waveresponse.Grid.interpolate`
method:

.. code-block:: python

    freq_new = np.array([0, 0.5, 1.0])
    dirs_new = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    vals_new = grid.interpolate(freq_new, dirs_new, freq_hz=True, degrees=True)

The underlying coordinate system can be rotated:

.. code-block:: python

    grid_rot = grid.rotate(45.0, degrees=True)

Or reshaped to match some other frequency/direction coordinates. Then, the values
are interpolated to match those new coordinates.

.. code-block:: python

    freq_new = np.array([0, 0.5, 1.0])
    dirs_new = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    grid_reshaped = grid.reshape(freq_new, dirs_new, freq_hz=True, degrees=True)

Some basic arithmetics and mathematical operations are provided. These operations
will be done on the grid's values (2-D array).

.. code-block:: python

    # Multiply
    grid_mul = grid * grid
    grid_mul_scalar = 2. * grid

    # Add
    grid_added = grid + grid
    grid_added_scalar = grid + 2.

    # Subtract
    grid_sub = grid - grid
    grid_sub_scalar = 1. - grid

    # Convert to real or imaginary parts
    grid_real = grid.real
    grid_imag = grid.imag
