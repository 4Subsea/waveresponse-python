DirectionalBinSpectrum
======================
The :class:`~waveresponse.DirectionalBinSpectrum` class provides an interface for
handling 2-D directional spectra. :class:`~waveresponse.DirectionalBinSpectrum`
extends :class:`~waveresponse.Grid`, and contains spectrum density as a function
of frequency, binned by direction.

.. math::
    \sum_{i=0}^n{S_i(\omega, \delta\left(\theta - \theta_i\right))}

The :class:`~waveresponse.DirectionalBinSpectrum` is initialized with a frequency
list (1-D array), a direction list (1-D array) and corresponding spectrum
density values, binned by direction (2-D array).

.. code-block:: python

    import numpy as np
    from waveresponse as wr


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    spectrum = wr.DirectionalBinSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=False,
    )

The :class:`~waveresponse.DirectionalBinSpectrum` class extends the :class:`~waveresponse.Grid`
class with the following:

Calculate the variance (i.e., integral) and standard deviation of the spectrum:

.. code-block:: python

    # Variance
    var = spectrum.var()

    # Standard deviation
    std = spectrum.std()

Integrate (or sum) over one of the axes to obtain a one-dimentional spectrum.
You can specify whether to integrate over the frequency axis (``axis=0``), or
sum over the direction axis (``axis=1``), by setting the appropriate `axis` parameter.

.. code-block:: python

    # "Non-directional" spectrum
    spectrum_nondir = spectrum.spectrum1d(axis=1)

    # Directional "histogram"
    spectrum_dir = spectrum.spectrum1d(axis=0)

Calculate spectral moments by calling the :meth:`~waveresponse.DirectionalBinSpectrum.moment`
method with the desired order, `n`.

.. code-block:: python

    # Zeroth-order moment
    m0 = spectrum.moment(0)

    # First-order moment
    m1 = spectrum.moment(1)

    # Second-order moment
    m2 = spectrum.moment(2)

    # Etc.

Calculate the mean zero-crossing period, Tz:

.. code-block:: python

    spectrum.tz

Calculate extreme values using the :meth:`~waveresponse.DirectionalSpectrum.extreme`
method. The method takes three arguments: the duration of the process (in seconds),
the quantile, ``q``, and a boolean flag, ``absmax``, determining whether to compute absolute
value extremes (or only consider the maxima (`default`)).

.. code-block:: python

    duration = 3 * 3600   # 3 hours

    # Extreme maximum
    mpm = spectrum.extreme(duration, q=0.37)   # most probable maximum (MPM)
    q90 = spectrum.extreme(duration, q=0.90)   # 90-th quantile

    # Extreme absolute value maximum (i.e., minima are taken into account)
    mpm = spectrum.extreme(duration, q=0.37, absmax=True)   # most probable maximum (MPM)
    q90 = spectrum.extreme(duration, q=0.90, absmax=True)   # 90-th quantile
