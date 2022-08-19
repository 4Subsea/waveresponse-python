DirectionalSpectrum
===================
The :class:`~waveresponse.DirectionalSpectrum` class provides an interface for
handling 2-D directional spectra. :class:`~waveresponse.DirectionalSpectrum`
extends :class:`~waveresponse.Grid`, and contains spectrum density values
on a two-dimentional frequency/(wave)direction grid.

The :class:`~waveresponse.DirectionalSpectrum` is initialized with a frequency
list (1-D array), a direction list (1-D array) and corresponding spectrum density
values (2-D array).

.. code-block:: python

    import numpy as np
    from waveresponse as wr


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    spectrum = wr.DirectionalSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
        clockwise=False,
        waves_coming_from=False,
    )

The :class:`~waveresponse.DirectionalSpectrum` class extends the :class:`~waveresponse.Grid`
class with the following:

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
    spectrum_nondir = spectrum.spectrum1d(axis=1)

    # Directional "distribution"
    spectrum_dir = spectrum.spectrum1d(axis=0)

Calculate spectral moments by calling the :meth:`~waveresponse.DirectionalSpectrum.moment`
method with the desired order, `n`.

.. code-block:: python

    # Zeroth-order moment
    m0 = spectrum.moment(0)

    # First-order moment
    m1 = spectrum.moment(1)

    # Second-order moment
    m2 = spectrum.moment(2)

    # Etc.
