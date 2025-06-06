.. _calculate-response:

Calculate response spectrum
===========================

This section will show you how to calculate a wave-induced first-order motion response
spectrum using ``waveresponse``. Linear theory and stationary conditions are assumed;
then, the vessel's frequency-domain response is described by a (motion) response
amplitude operator (RAO) and a 2-D wave spectrum. Calculation of the response spectrum
is governed by the following equation:

.. math::
    S_x(\omega) = \int H_x(\omega, \theta)H_x^{*}(\omega, \theta) S_{\zeta}(\omega, \theta) d\theta

where :math:`S_{\zeta}(\omega, \theta)` is the 2-D wave spectrum, :math:`H_x(\omega, \theta)`
is the degree-of-freedom's transfer function (i.e., RAO), :math:`H_x^{*}(\omega, \theta)`
is the complex conjugate version of the transfer function, and :math:`S_x(\omega)`
is the response spectrum. :math:`\theta` is the relative wave direction, and :math:`\omega`
is the angular frequency.

.. note::
    Keep in mind that the wave spectrum and the RAO must be given in compatible
    units when calculating response according to the above equation. E.g., for
    roll (:math:`\alpha`):

    .. math::
        S_{\alpha}(\omega) \left[\frac{rad^2}{Hz}\right] = \int_0^{2\pi} H_{\alpha}(\omega, \theta) \left[\frac{rad}{m}\right] \cdot H_{\alpha}^{*}(\omega, \theta) \left[\frac{rad}{m}\right] \cdot S_{\zeta}(\omega, \theta) \left[\frac{m^2}{Hz \cdot rad}\right] d\theta

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

    import waveresponse as wr


    def calculate_response(rao, wave, heading, heading_degrees=False):
        """
        Calculate response spectrum.

        Parameters
        ----------
        rao : RAO
            Response amplitude operator (RAO).
        wave : WaveSpectrum
            Wave spectrum.
        heading : float
            Heading of vessel relative to wave spectrum reference frame.
        heading_degrees : bool
            Whether the heading is given in 'degrees'. If ``False``, 'radians' is assumed.

        Returns
        -------
        DirectionalSpectrum :
            Response spectrum.
        """

        # Rotate wave spectrum to 'body' frame
        wave_body = wave.rotate(heading, degrees=heading_degrees)

        # Ensure that ``rao`` and ``wave`` has the same 'wave convention'
        wave_body.set_wave_convention(**rao.wave_convention)

        # Reshape the RAO to match the wave spectrum's frequency/direction coordinates.
        # It is recommended to reshape (i.e., interpolate) the magnitude-squared
        # version of the RAO when estimating response, since this has shown best
        # results:
        #    https://cradpdf.drdc-rddc.gc.ca/PDFS/unc341/p811241_A1b.pdf
        freq = wave_body.freq(freq_hz=False)
        dirs = wave_body.dirs(degrees=False)
        rao_squared = (rao * rao.conjugate()).real
        rao_squared = rao_squared.reshape(freq, dirs, freq_hz=False, degrees=False)

        return wr.multiply(rao_squared, wave_body, output_type="DirectionalSpectrum")

The response is returned as a :class:`~waveresponse.DirectionalSpectrum` object,
and provides useful spectrum operations, such as:

.. code-block:: python

    # Integrate over direction to get the 'non-directional' response spectrum
    freq, response_spectrum = response.spectrum1d(axis=1)

    # Calculate response variance
    var = response.var()

    # Calculate response standard deviation
    std = response.std()

    # Etc.

.. note::

    :meth:`~waveresponse.calculate_response` returns the response as a two-dimentional
    spectrum calculated according to:\

    .. math::
        S_x(\omega, \theta) = H_x(\omega, \theta)H_x^{*}(\omega, \theta) S_{\zeta}(\omega, \theta)

    To obtain the one-dimentional spectrum, you need to integrate over direction:

    .. math::
        S_x(\omega) = \int S_x(\omega, \theta) d\theta

    The response spectrum does not make much physical sense before it is integrated
    and converted to a 1-D non-directional spectrum. However, the 2-D version can
    indicate which wave directions are most important for the total response.
