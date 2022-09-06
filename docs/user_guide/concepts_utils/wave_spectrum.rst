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

.. note::
    Often you do not have access to a full directional wave spectrum. Then, it is
    common to instead construct a directional spectrum from a standardized frequency
    spectrum, :math:`S(\omega)`, and a directional spreading function,
    :math:`D(\omega, \theta)`:

    .. math::
        S(\omega, \theta) = S(\omega) D(\omega, \theta)

    Since the frequency spectrum is obtained by integrating
    the directional spectrum over the directional domain (i.e., [0, 360)  degrees,
    or [0, 2\ :math:`\pi`) radians),

    .. math::
        S(\omega) = \int_0^{2\pi} S(\omega, \theta)

    we get the following requirement for the spreading function for each frequency,
    :math:`\omega_i`:

    .. math::
        \int_0^{2\pi} D(\omega_i, \theta) = 1

    In general, the spreading function is a function of both frequency, :math:`\omega`,
    and direction, :math:`\theta`. However, it is common to use the same spreading
    for all frequencies. Standardized spreading functions (denoted :math:`\kappa` here), are usually
    defined such that they have their maximum value at :math:`\theta = 0`. From these
    standardized spreading functions, we can obtain a spreading function with an
    arbitrary peak direction, :math:`\theta_p`, by:

    .. math::
        D(\omega, \theta) = \kappa(\theta - \theta_p)

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


Standardized wave spectra
-------------------------
Often you 
