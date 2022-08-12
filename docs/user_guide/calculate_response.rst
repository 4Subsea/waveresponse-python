Calculate response spectrum
===========================

This section will show you how to calculate a first-order wave-induced motion response
spectrum using ``scarlet_lithium``. We will assume linear theory, where a vessel's
response is sufficiently described by a (motion) response amplitude operator (RAO)
and a 2-D wave spectrum. The response spectrum estimation is governed by:

.. math::
    S_x(\omega) = \int H_x(\omega, \beta)H_x^{*}(\omega, \beta) S_{\zeta}(\omega, \beta) d\beta

where :math:`S_{\zeta}(\omega, \beta)` is the 2-D wave spectrum, :math:`H_x(\omega, \beta)`
is the degree-of-freedom's transfer function (i.e., RAO), :math:`H_x^{*}(\omega, \beta)`
is the complex conjugate version of the transfer function, and :math:`S_x(\omega)`
is the response spectrum. :math:`\beta` is the relative wave direction.

With ``scarlet_lithium`` it is easy to estimate a vessel's response spectrum once
you have an :class:`~scarlet_lithium.RAO` object and a :class:`~scarlet_lithium.WaveSpectrum`
object available. A convenience function for calculating response is provided by
:func:`~scarlet_lithium.calculate_response`:

.. code-block:: python

    from scarlet_lithium import calculate_response


    heading = 45.0   # degrees
    response = calculate_response(wave, rao, heading, heading_degrees=True)

This function is roughly equivalent to:

.. code-block:: python

    def calculate_response(wave, rao, heading, heading_degrees=False):
        """
        Calculate response spectrum.

        Parameters
        ----------
        wave : obj
            ``WaveSpectrum`` object.
        rao : obj
            ``RAO`` object.
        heading : float
            Heading of vessel relative to wave spectrum coordinate system.
        heading_degrees : bool
            Whether the heading is given in 'degrees'. If ``False``, 'radians' is assumed.

        Returns
        -------
        obj :
            Response spectrum.
        """

        # Rotate wave spectrum to 'body' frame
        wave_body = wave.rotate(heading, degrees=heading_degrees)

        # Ensure that ``rao`` and ``wave`` has the same 'wave convention'
        wave_body.set_wave_convention(**rao.wave_convention)

        # Reshape ``rao`` and ``wave`` so that they share the same frequency/direction
        # coordinates. In this example, ``wave`` will dictate the coordinates, and
        # the ``rao`` object will be interpoated to match these coordinates.
        # 
        # It is recommended to reshape (i.e., interpolate) the magnitude-squared
        # version of the RAO when estimating response, since this has shown best
        # results:
        #    https://cradpdf.drdc-rddc.gc.ca/PDFS/unc341/p811241_A1b.pdf
        freq = wave.freq(freq_hz=False)
        dirs = wave.dirs(degrees=False)
        rao_squared = np.abs(rao * rao.conjugate())
        rao_squared = rao_squared.reshape(freq, dirs, freq_hz=False, degrees=False)
        wave_body = wave_body.reshape(freq, dirs, freq_hz=False, degrees=False)

        return rao_squared * wave_body

The response is returned as a :class:`~scarlet_lithium.DirectionalSpectrum` object,
and provides useful spectrum operations, such as:

.. code-block:: python

    # Get 'non-directional' response spectrum
    freq, response_spectrum = response.spectrum1d(axis=1)

    # Calculate response variance
    var = response.var()

    # Calculate response standard deviation
    std = response.std()

    # Etc.
