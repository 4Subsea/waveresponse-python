DirectionalSpectrum
===================
The :class:`~scarlet_lithium.DirectionalSpectrum` class provides an interface for
handling 2-D directional spectra. :class:`~scarlet_lithium.DirectionalSpectrum`
inherits from :class:`~scarlet_lithium.Grid`, and contains spectrum density values
on a two-dimentional frequency/(wave)direction grid. :class:`~scarlet_lithium.DirectionalSpectrum`
is the base class for :class:`~scarlet_lithium.WaveSpectrum`.

The :class:`~scarlet_lithium.DirectionalSpectrum` is initialized with a frequency
list (1-D array), a direction list (1-D array) and corresponding spectrum density
values (2-D array).

.. code-block:: python

    import numpy as np
    from scarlet_lithium import DirectionalSpectrum


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    spectrum = DirectionalSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=False,
    )

The :class:`~scarlet_lithium.DirectionalSpectrum` class inherits from the :class:`~scarlet_lithium.Grid`
class, and provides all the functionality that comes with :class:`~scarlet_lithium.Grid`.
In addition, you can:

Calculate the variance (i.e., integral) and standard deviation of the spectrum:

.. code-block:: python

    # Variance
    var = spectrum.var()

    # Standard deviation
    std = spectrum.std()

Integrate over one of the axes to obtain a one-dimentional spectrum. You can specify
whether to integrate over the frequency axis (``axis=0``), or the direction axis
(``axis=1``), by setting the appropriate `axis` parameter.

.. code-block:: python

    # "Non-directional" spectrum
    spectrum_nondir = spectrum1d(axis=1)

    # Directional "distribution"
    spectrum_dir = spectrum1d(axis=0)

Calculate spectral moments by calling the :meth:`~scarlet_lithium.DirectionalSpectrum.moment`
method with the desired order, `n`.

.. code-block:: python

    # Zeroth-order moment
    m0 = spectrum.moment(0)

    # First-order moment
    m1 = spectrum.moment(1)

    # Second-order moment
    m2 = spectrum.moment(2)

    # Etc.
