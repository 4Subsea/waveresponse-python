Grid
====
The :class:`~waveresponse.Grid` class provides an interface for handling 2-D
arrays of values that should be interpreted on two-dimentional frequency/(wave)direction
grid.

:class:`~waveresponse.Grid` provides some useful functionality on its own, but
the main intension of this class is to serve as a building block for other classes.
:class:`~waveresponse.RAO`, :class:`~waveresponse.DirectionalSpectrum` and
:class:`~waveresponse.WaveSpectrum` are all examples of classes that inherits
from :class:`~waveresponse.Grid`.

.. code-block:: python

    import numpy as np
    from waveresponse import Grid


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, 10, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    grid = Grid(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
    )

To be able to interpret the grid directions and values, we need some information
about the 'wave convention'. Two boolean parameters are needed:

*clockwise*
    Describes the direction of positive rotation.

*waves_coming_from*
    Describes the direction in which the waves are propagating.

These parameters are set during initialization of the grid object:

.. code-block:: python

    grid = Grid(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=True,
    )

If you want to convert the grid to a different wave convention, you can do that
by calling the :meth:`~waveresponse.Grid.set_wave_convention` method.

.. code-block:: python

    new_convention = {"clockwise": False, "waves_coming_from": True}
    grid.set_wave_convention(new_convention)

Once you have an initialized grid object, the grid's frequency/direction coordinates
and values can be retrieved by calling the grid:

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
