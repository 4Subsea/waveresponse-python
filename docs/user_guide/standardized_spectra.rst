Standardized wave spectra
=========================

Idealized 1-D spectra
#####################

Pierson-Moskowitz spectrum
--------------------------
The Pierson-Moskowits (PM) spectrum, :math:`S_{PM}(\omega)`, is given by:

.. math::

    S_{PM}(\omega) = \frac{A}{w^5} exp\left(-\frac{B}{\omega^4}\right)

where :math:`A` and :math:`B` are parameters that describe the shape of the spectrum.


Modified Pierson-Moskowitz spectrum
-----------------------------------
The modified Pierson-Moskowits (i.e., Bretschneider) spectrum, :math:`S_{PM}(\omega)`, is given by:

.. math::

    S_{PM} = \frac{5}{16}H_s^2\omega_p^2\omega^{-5} exp\left(-\frac{5}{4} \left( \frac{\omega_p}{\omega} \right)^4 \right)

where :math:`H_s` is the significant wave height and :math:`\omega_p = \frac{2\pi}{Tp}` is the
angular spectral peak frequency.

The :class:`~waveresponse.ModifiedPiersonMoskowitz` class provides functionality
for generating 1-D (modified) Pierson-Moskowitz spectra for given Hs/Tp combinations:

.. code:: python

    import numpy as np
    import waveresponse as wr


    freq = np.arange(0.01, 1, 0.1)   # Hz
    spectrum = wr.ModifiedPiersonMoskowitz(freq, freq_hz=True)

    hs = 3.5
    tp = 10.0
    freq, vals = spectrum(hs, tp)


JONSWAP spectrum
----------------
The JONSWAP spectrum, :math:`S_{JW}(\omega)`, is given by:

.. math::

    S_{JW} = \alpha_{\gamma}S_{PM}(\omega)\gamma^{exp\left( -\frac{(\omega - \omega_p)^2}{2\sigma^2\omega_p^2} \right)}

where,

- :math:`S_{PM}(w)` is the Pierson-Moskowitz (PM) spectrum.
- :math:`\gamma` is a peak enhancement factor.
- :math:`\alpha_{\gamma} = 1 - 0.287 \cdot ln(\gamma)` is a normalizing factor.
- :math:`\sigma` is the spectral width parameter:
    - :math:`\sigma = \sigma_a`, for :math:`\omega <= \omega_p`
    - :math:`\sigma = \sigma_b`, for :math:`\omega > \omega_p`
- :math:`\omega_p = \frac{2\pi}{Tp}` is the angular spectral peak frequency.

The :class:`~waveresponse.JONSWAP` class provides functionality for generating 1-D
JONSWAP spectra for a given Hs/Tp combinations:

.. code:: python

    import numpy as np
    import waveresponse as wr


    freq = np.arange(0.01, 1, 0.1)   # Hz
    spectrum = wr.JONSWAP(freq, freq_hz=True, gamma=2, sigma_a=0.07, sigma_b=0.09)

    hs = 3.5
    tp = 10.0
    freq, vals = spectrum(hs, tp)


Directional spectrum
####################
The directional spectrum is usually standardized in a similar way as the 1-D frequency
spectrum. The standardization is based on expressing the directional spectrum as
a product of a frequency spectrum, :math:`S(\omega)`, and a directional spreading
function, :math:`D(\theta, \omega)`:

.. math::
    S(\omega, \theta) = S(\omega) D(\theta, \omega)

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
for all frequencies.


Cosine-2s based spreading
-------------------------
Standardized spreading functions (denoted :math:`\kappa(\hat{\theta})` here),
are usually defined such that they have their maximum value at :math:`\hat{\theta} = 0`.
From these standardized spreading functions, we can obtain a spreading function
with an arbitrary peak direction, :math:`\theta_p`, by taking:

.. math::

    D(\omega, \theta) = \kappa(\theta - \theta_p)

Cosine-based spreading functions are most common. ``waveresponse`` provides two
variations of the cosine-based spreading: one that spreads the wave energy over
the full directional domain, and one that spreads the energy over half the domain.

The :class:`~waveresponse.CosineFullSpreading` class provides directional spreading
according to:

.. math::

    \kappa(\hat{\theta}) = \frac{2^{2s-1}}{\pi} \frac{\Gamma^2(s+1)}{\Gamma^2(2s+1)} cos^{2s} \left(\frac{\hat{\theta}}{2}\right)


The :class:`~waveresponse.CosineHalfSpreading` class provides directional spreading
according to:

.. math::

    \kappa(\hat{\theta}) = \Gamma(a, b, c)
