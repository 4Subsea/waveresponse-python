DirectionalSpectrum
===================
This is how the :class:`~scarlet_lithium.DirectionalSpectrum` class works.

The :class:`~scarlet_lithium.DirectionalSpectrum` is initialized with a frequency
list (1-D array), a direction list (1-D array) and corresponding spectrum density
values (2-D array).

.. code-block:: python

    import numpy as np
    from scarlet_lithium import DirectionalSpectrum


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals = np.random.random((len(freq), len(dirs)))

    rao = DirectionalSpectrum(
        freq,
        dirs,
        vals,
        freq_hz=True,
        degrees=True,
    )
