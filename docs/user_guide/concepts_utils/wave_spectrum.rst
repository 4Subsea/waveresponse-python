WaveSpectrum
============
The :class:`~scarlet_lithium.WaveSpectrum` class provides an interface for handling
2-D directional wave spectra. :class:`~scarlet_lithium.WaveSpectrum` inherits from
:class:`~scarlet_lithium.Grid`, and contains spectrum density values on a two-dimentional
frequency/(wave)direction grid.

The :class:`~scarlet_lithium.WaveSpectrum` is initialized with a frequency
list (1-D array), a direction list (1-D array) and corresponding wave spectrum density
values (2-D array).

.. code-block:: python

    import numpy as np
    from scarlet_lithium import WaveSpectrum


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    wave = WaveSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=False,
    )

The :class:`~scarlet_lithium.WaveSpectrum` class inherits from the
:class:`~scarlet_lithium.DirectionalSpectrum` class (and the :class:`~scarlet_lithium.Grid`
class), and provides all the functionality that comes with
:class:`~scarlet_lithium.DirectionalSpectrum` and :class:`~scarlet_lithium.Grid`.
In addition, you can:

Calculate the significant wave height, Hs:

.. code-block:: python

    wave.hs

Calculate the wave peak period, Tp:

.. code-block:: python

    wave.tp

Calculate the mean crossing period, Tz:

.. code-block:: python

    wave.tz

Calculate the wave peak direction:

.. code-block:: python

    wave.dirp()

Calculate the mean wave direction:

.. code-block::

    wave.dirm()
