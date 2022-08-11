Calculate response spectrum
===========================
With ``scarlet_lithium`` it is easy to estimate a vessel's response spectrum once
you have a :class:`~scarlet_lithium.RAO` object and a :class:`~scarlet_lithium.WaveSpectrum`
object available.

A convenience function for calculating response is provided by
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
        # coordinates.
        # It is recommended to reshape (i.e., interpolate) the magnitude-squared
        # version of the RAO when estimating the response, since this has shown
        # best results:
        #    https://cradpdf.drdc-rddc.gc.ca/PDFS/unc341/p811241_A1b.pdf
        freq_ = rao._freq
        dirs_ = wave._dirs
        rao_squared = np.abs(rao * rao.conjugate())
        rao_squared = rao_squared.reshape(freq_, dirs_, freq_hz=False, degrees=False)
        wave_body = wave_body.reshape(freq_, dirs_, freq_hz=False, degrees=False)

        return rao_squared * wave_body

The response is returned as a :class:`~scarlet_lithium.DirectionalSpectrum` object,
and provides useful spectrum operations such as:

.. code-block:: python

    # Get non-directional response spectrum
    spectrum1d = response.spectrum1d(axis=1)

    # Calculate response variance
    var = response.var()

    # Calculate response standard deviation
    std = response.std()

    # Etc.
