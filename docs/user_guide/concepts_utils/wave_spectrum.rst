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
spectrum (1-D array), a spreading function and a peak direction:

.. code-block:: python

    import numpy as np
    import waveresponse as wr


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    spectrum1d = np.random.random(len(freq))
    dirp = 45.0

    spread_fun = lambda f, d: (1.0 / 180.0) * np.cos(np.radians(d / 2)) ** 2

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

.. note::
    The directional spectrum is constructed according to,

    .. math::
        S(f, \beta) = S(f) D(f, \beta - \beta_p)

    where :math:`S(f)` is the non-directional spectrum, :math:`D(f, \beta - \beta_p)` is
    the spreading function (sometimes referred to as the 'directional distribution'),
    and :math:`\beta_p` is the spectrum's peak direction. In general, the spreading
    function should be a function of both frequency, :math:`f`, and direction,
    :math:`\beta`. However, it is common practice to use
    the same spreading for all frequencies.

    The spreading function must be defined such that it for each frequency, :math:`f_i`, yields unity
    integral over the direction domain (i.e., [0, 360) degrees, or [0, numpy.pi)):

    .. math::
        \int_0^{2\pi} D(f_i, \beta) d\beta = 1

    The spreading function should have its maximum value at :math:`\beta - \beta_p = 0`.

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
