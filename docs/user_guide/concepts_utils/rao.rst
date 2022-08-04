RAO
===
The :class:`~scarlet_lithium.RAO` class provides an interface for handling response
amplitude operators (RAOs). :class:`~scarlet_lithium.RAO` inherits from :class:`~scarlet_lithium.Grid`,
and contains RAO values on a two-dimentional frequency/(wave)direction grid.

The :class:`~scarlet_lithium.RAO` is initialized with a frequency list (1-D array),
a direction list (1-D array) and corresponding RAO values as complex numbers (2-D array).

.. code-block:: python

    import numpy as np
    from scarlet_lithium import RAO


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    vals_real = np.random.random((len(freq), len(dirs)))
    vals_imag = np.random.random((len(freq), len(dirs)))
    vals_complex = vals_real + 1j * vals_imag

    rao = RAO(
        freq,
        dirs,
        vals_complex,
        freq_hz=True,
        degrees=True,
    )

Alternatively, you can construct an :class:`~scarlet_lithium.RAO` using amplitudes
(2-D array) and phase (2-D array):

.. code-block:: python

    import numpy as np
    from scarlet_lithium import RAO


    freq = np.linspace(0.0, 1.0, 50)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    amp = np.random.random((len(freq), len(dirs)))
    phase = np.random.random((len(freq), len(dirs)))

    rao = RAO.from_amp_phase(
        freq,
        dirs,
        amp,
        phase,
        phase_degrees=False,
        freq_hz=True,
        degrees=True,
    )

The :class:`~scarlet_lithium.RAO` class inherits from the :class:`~scarlet_lithium.Grid`
class, and provides all the functionality that comes with :class:`~scarlet_lithium.Grid`.
In addition, you can:

Retrieve the RAO's frequency/direction coordinates and amplitude/phase values using
:meth:`~scarlet_lithium.RAO.to_amp_phase`.

.. code-block:: python

    freq, dirs, amp, phase = rao.to_amp_phase(freq_hz=True, degrees=True)


Get the complex conjugated version of the RAO using :meth:`~scarlet_lithium.RAO.conjugate`.

.. code-block:: python

    rao_conj = rao.conjugate()

Differentiate to obtain an RAO object that represents the differentiated degree-of-freedom:

.. code-block:: python

    rao_diff = rao.differentiate()

.. note::
    The differentiated version of an RAO's transfer function is obtained by:

    .. math::

        j\omega H(j\omega)