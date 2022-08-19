WaveSpectrum
============
The :class:`~waveresponse.WaveSpectrum` class provides an interface for handling
2-D directional wave spectra. :class:`~waveresponse.WaveSpectrum` extends
:class:`~waveresponse.DirectionalSpectrum`, and contains spectrum density values on
a two-dimensional frequency/(wave)direction grid.

The :class:`~waveresponse.WaveSpectrum` is initialized with a frequency
list (1-D array), a direction list (1-D array) and corresponding wave spectrum density
values (2-D array).

.. code-block:: python

    import numpy as np
    from waveresponse as wr


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    wave = wr.WaveSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=False,
    )

The :class:`~waveresponse.WaveSpectrum` extends the
:class:`~waveresponse.DirectionalSpectrum` class with the following:

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
