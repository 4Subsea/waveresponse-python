Grid
====
This is how the :class:`~scarlet_lithium.Grid` class works.

.. code-block:: python

    import numpy as np
    from scarlet_lithium import Grid

    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals = np.ones((len(freq), len(dirs)))

    grid = Grid(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
    )

The grid's frequency/direction coordinates and values can be retrieved by calling
the grid object:

.. code-block:: python

    freq, dirs, vals = grid(freq_hz=True, degrees=True)

Interpolation of the grid values is provided by the :meth:`~scarlet_lithium.Grid.interpolate`
method:

.. code-block:: python

    freq_new = np.array([0, 0.5, 1.0])
    dirs_new = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    vals_new = grid.interpolate(freq_new, dirs_new, freq_hz=True, degrees=True)

The underlying coordinate system can be rotated:

.. code-block:: python

    grid_rot = grid.rotate(45.0, degrees=True)

Or reshaped to match some new frequency/direction coordinates. Then, the values
are interpolated to match those new coordinates.

.. code-block:: python

    freq_new = np.array([0, 0.5, 1.0])
    dirs_new = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    grid_reshaped = grid.reshape(freq_new, dirs_new, freq_hz=True, degrees=True)


Wave convention
---------------

The :class:`~scarlet_lithium.Grid` class is used to handle values that are related
to wave direction (and frequency). To be able to interpret the grid values, we need
some information about the 'wave direction convention'. Two boolean parameters are
needed:

*clockwise*
    Describes the direction of positive rotation.

*waves_coming_from*
    Describes the direction in which the waves are propagating.

The 'wave convention' that is set, can be accessed through the :attr:`~scarlet_lithium.Grid.wave_convention`
property. If you want to convert the grid values to a different convention, you
can do that by calling the :meth:`~scarlet_lithium.Grid.set_wave_convention` method.

.. code-block:: python

    # Get the wave convention
    wave_convention = grid.wave_convention

    # Change the wave convention (converts all wave directions to the new convention)
    new_convention = {"clockwise": False, "waves_coming_from": True}
    grid.set_wave_convention(new_convention)
