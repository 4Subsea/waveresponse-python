WaveBinSpectrum
===============
The :class:`~waveresponse.WaveBinSpectrum` class provides an interface for handling
2-D directional wave spectra. :class:`~waveresponse.WaveSpectrum` extends
:class:`~waveresponse.DirectionalBinSpectrum`, and contains spectrum density as a function
of frequency, binned by direction.

.. math::
    \sum_{i=0}^n{S_{\zeta, i}(\omega, \delta\left(\theta - \theta_i\right))}

The :class:`~waveresponse.WaveSpectrum` is initialized with a frequency
list (1-D array), a direction list (1-D array) and corresponding spectrum
density values, binned by direction (2-D array).

.. code-block:: python

    import numpy as np
    import waveresponse as wr


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    wave = wr.WaveBinSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=False,
    )


The :class:`~waveresponse.WaveBinSpectrum` class extends the
:class:`~waveresponse.DirectionalBinSpectrum` class with the following:

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
