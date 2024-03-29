WaveSpectrum
============
The :class:`~waveresponse.WaveSpectrum` class provides an interface for handling
2-D directional wave spectra. :class:`~waveresponse.WaveSpectrum` extends
:class:`~waveresponse.DirectionalSpectrum`, and contains spectrum density values on
a two-dimensional frequency/(wave)direction grid.

.. math::
    S_{\zeta}(\omega, \theta)

The :class:`~waveresponse.WaveSpectrum` is initialized with a frequency
list (1-D array), a direction list (1-D array) and corresponding wave spectrum density
values (2-D array).

.. code-block:: python

    import numpy as np
    import waveresponse as wr


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

Alternatively, you can construct a :class:`~waveresponse.WaveSpectrum` from a 'non-directional'
spectrum (1-D array), a directional spreading function and a peak direction:

.. code-block:: python

    import numpy as np
    import waveresponse as wr


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    spectrum1d = np.random.random(len(freq))
    dirp = 45.0

    def spread_fun(f, d):
        return (1.0 / 180.0) * np.cos(np.radians(d / 2)) ** 2

    wave = wr.WaveSpectrum.from_spectrum1d(
        freq,
        dirs,
        spectrum1d,
        spread_fun,
        dirp,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=False,
    )

.. tip::
    Two standardized (cosine-based) spreading functions, :class:`~waveresponse.CosineFullSpreading`
    and :class:`~waveresponse.CosineHalfSpreading`, are provided by ``waveresponse``.

The :class:`~waveresponse.WaveSpectrum` class extends the
:class:`~waveresponse.DirectionalSpectrum` class with the following:

Calculate the significant wave height, Hs:

.. code-block:: python

    wave.hs

Calculate the wave peak period, Tp:

.. code-block:: python

    wave.tp

Calculate the wave peak direction:

.. code-block:: python

    wave.dirp()

Calculate the mean wave direction:

.. code-block::

    wave.dirm()
