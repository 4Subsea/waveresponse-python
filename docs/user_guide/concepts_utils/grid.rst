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
    from waveresponse as wr


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

To be able to interpret the (wave) directions and values that are associated with
a grid, we need some information about the 'wave convention' that is used. Two boolean
parameters are needed:

*clockwise*
    Describes the direction of positive rotation.

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

If you want to convert the grid to a different wave convention, you can achieve
that by calling the :meth:`~waveresponse.Grid.set_wave_convention` method with the
desired convention flags.

.. code-block:: python

    grid.set_wave_convention(clockwise=False, waves_coming_from=True)

Once you have an initialized grid object, the grid's frequency/direction coordinates
and values can be retrieved by calling the grid. You must then specify which coordinate
units to return by setting the ``freq_hz`` and ``degrees`` flags.

.. code-block:: python

    freq, dirs, vals = grid(freq_hz=True, degrees=True)

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
    grid_squared = grid * grid

    # Convert to absolute values
    grid_abs = np.abs(grid)

    # Convert to real or part
    grid_real = np.real(grid)
    grid_imag = np.imag(grid)
