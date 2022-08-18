.. _calculate-response:

Calculate response spectrum
===========================

This section will show you how to calculate a wave-induced first-order motion response
spectrum using ``waveresponse``. Linear theory and stationary conditions are assumed;
then, the vessel's frequency-domain response is described by a (motion) response
amplitude operator (RAO) and a 2-D wave spectrum. Calculation of the response spectrum
is governed by the following equation:

.. math::
    S_x(\omega) = \int H_x(\omega, \beta)H_x^{*}(\omega, \beta) S_{\zeta}(\omega, \beta) d\beta

where :math:`S_{\zeta}(\omega, \beta)` is the 2-D wave spectrum, :math:`H_x(\omega, \beta)`
is the degree-of-freedom's transfer function (i.e., RAO), :math:`H_x^{*}(\omega, \beta)`
is the complex conjugate version of the transfer function, and :math:`S_x(\omega)`
is the response spectrum. :math:`\beta` is the relative wave direction, and :math:`\omega`
is the angular frequency.

With ``waveresponse`` it is easy to estimate a vessel's response spectrum once
you have an :class:`~waveresponse.RAO` object and a :class:`~waveresponse.WaveSpectrum`
object available. A convenience function for calculating response is provided by
:func:`~waveresponse.calculate_response`:

.. code-block:: python

    import waveresponse as wr


    heading = 45.0   # degrees
    response = wr.calculate_response(rao, wave, heading, heading_degrees=True)

This function is roughly equivalent to:

.. code-block:: python

    def calculate_response(rao, wave, heading, heading_degrees=False):
        """
        Calculate response spectrum.

        Parameters
        ----------
        rao : obj
            ``RAO`` object.
        wave : obj
            ``WaveSpectrum`` object.
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
        freq = wave_body.freq(freq_hz=False)
        dirs = wave_body.dirs(degrees=False)
        rao_squared = np.abs(rao * rao.conjugate())
        rao_squared = rao_squared.reshape(freq, dirs, freq_hz=False, degrees=False)
        wave_body = wave_body.reshape(freq, dirs, freq_hz=False, degrees=False)

        return rao_squared * wave_body

The response is returned as a :class:`~waveresponse.DirectionalSpectrum` object,
and provides useful spectrum operations, such as:

.. code-block:: python

    # Integrate over direction, and get the 'non-directional' response spectrum
    freq, response_spectrum = response.spectrum1d(axis=1)

    # Calculate response variance
    var = response.var()

    # Calculate response standard deviation
    std = response.std()

    # Etc.


.. note::

    :meth:`~waveresponse.calculate_response` returns the response as a 2-D spectrum
    calculated according to:\

    .. math::
        H_x(\omega, \beta)H_x^{*}(\omega, \beta) S_{\zeta}(\omega, \beta)

    To obtain the one-dimentional spectrum (which is what you would measure),
    you need to integrate over direction.
