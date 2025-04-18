Standardized wave spectra
=========================

Idealized 1-D spectra
#####################
It is common practice to approximate the actual wave spectrum with a standardized wave spectrum (which there exists many of).
``waveresponse`` provides the most common standardized 1D wave spectra.


Pierson-Moskowitz
-----------------

Pierson-Moskowitz (PM) type spectra has the following form:

.. math::

    S_{PM}(\omega) = \frac{A}{\omega^5} exp\left(-\frac{B}{\omega^4}\right)

Many of the most common standardized wave spectra today are of the PM type (or extends it). The
*modified Pierson-Moskowitz* spectrum and the *JONSWAP* spectrum are two examples.

.. It is common to express the spectrum parameters, :math:`A` and :math:`B`, in terms
.. of the significant wave height, Hs, and the wave peak period, Tp.


Modified Pierson-Moskowitz
..........................
The *modified Pierson-Moskowits* spectrum (also known as Bretschneider or ISSC) is given by:

.. math::

    S_{PM}(\omega) = \frac{5}{16}H_s^2\omega_p^4\omega^{-5} exp\left(-\frac{5}{4} \left( \frac{\omega_p}{\omega} \right)^4 \right)

where :math:`H_s` is the significant wave height and :math:`\omega_p = \frac{2\pi}{Tp}` is the
angular spectral peak frequency.

The :class:`~waveresponse.ModifiedPiersonMoskowitz` class provides functionality
for generating a 1-D (modified) Pierson-Moskowitz spectrum given two parameters (i.e.,
:math:`H_s` and :math:`T_p`):

.. code:: python

    import numpy as np
    import waveresponse as wr


    freq = np.arange(0.01, 1, 0.01)
    spectrum = wr.ModifiedPiersonMoskowitz(freq, freq_hz=True)

    hs = 3.5
    tp = 10.0
    freq, vals = spectrum(hs, tp)


JONSWAP
-------
The *JONSWAP* spectrum is given by:

.. math::

    S_{J}(\omega) = \alpha_{\gamma}S_{PM}(\omega)\gamma^{exp\left( -\frac{(\omega - \omega_p)^2}{2\sigma^2\omega_p^2} \right)}

where,

- :math:`S_{PM}(w)` is the modified Pierson-Moskowitz (PM) spectrum.
- :math:`\gamma` is a peak enhancement factor.
- :math:`\alpha_{\gamma}` is a normalizing factor.
- :math:`\omega_p = \frac{2\pi}{Tp}` is the angular spectral peak frequency.
- :math:`\sigma` is the spectral width parameter, given by:

.. math::
    \sigma =
    \begin{cases}
        \sigma_a & \quad \text{if } \omega \leq \omega_p\\
        \sigma_b & \quad \text{if } \omega > \omega_p
    \end{cases}

It is common to use :math:`\sigma_a = 0.07` and :math:`\sigma_b = 0.09` (established
from experimental data). Then, the normalizing factor becomes
:math:`\alpha_{\gamma} = 1 - 0.287 \cdot ln(\gamma)`, and the JONSWAP spectrum simplifies
to a three-parameter spectrum described by :math:`H_s`, :math:`T_p` and :math:`\gamma`.

The :class:`~waveresponse.JONSWAP` class provides functionality for generating a 1-D
JONSWAP spectrum given three parameters (i.e., :math:`H_s`, :math:`T_p` and :math:`\gamma`):

.. code:: python

    import numpy as np
    import waveresponse as wr


    freq = np.arange(0.01, 1, 0.01)
    spectrum = wr.JONSWAP(freq, freq_hz=True)

    hs = 3.5
    tp = 10.0
    freq, vals = spectrum(hs, tp, gamma=3.3)

.. note::

    For the special case where :math:`\gamma = 1`, JONSWAP corresponds to the modified Pierson-Moskowitz
    spectrum.

The JONSWAP spectrum is expected to be a reasonable model for:

.. math::

    3.6 \leq \frac{T_p}{\sqrt{H_s}} \leq 5


Ochi-Hubble
-----------
The *Ochi-Hubble* spectrum allows you to set up a double-peaked spectrum that represents
sea states which includes both a remotely generated swell component (with low frequency energy)
and a locally generated wind component (with high frequency energy). The Ochi-Hubble spectrum
is described by six parameters (three for each wave component), and is given by:

.. math::

    S_{OH}(\omega) = \frac{1}{4} \sum_j \frac{\left( \frac{4q_j+1}{4}\omega_{pj} \right)^{q_j}}{\Gamma(q_j)}
    \frac{H_{sj}^2}{\omega^{4q_j+1}}exp\left( -\frac{4q_j+1}{4} \left( \frac{\omega_{pj}}{\omega} \right)^4 \right)

where,

- :math:`H_{sj}` is the significant wave height for wave component :math:`j`.
- :math:`\omega_{pj} = \frac{2\pi}{T_{pj}}` is the angular spectral peak frequency for wave component :math:`j`.
- :math:`q_i` is a spectral shape parameter for wave component :math:`j`.

The index, :math:`j = 1, 2`, represents the lower frequency component (i.e., swell)
and higher frequency component (i.e., wind) respectively.

.. note::

    The Ochi-Hubble spectrum is implemented according to the `orinal paper <https://icce-ojs-tamu.tdl.org/icce/article/view/3066/2731>`_
    by M. K. Ochi and E. N. Hubble published in 1976. Refer to this paper for full implementation
    details.

The :class:`~waveresponse.OchiHubble` class provides functionality for generating a 1-D
Ochi-Hubble spectrum component given three parameters (i.e., :math:`H_s`, :math:`T_p`
and :math:`q`). The total spectrum is obtained by adding together the two wave components.

.. code:: python

    import numpy as np
    import waveresponse as wr


    freq = np.arange(0.01, 1, 0.01)
    spectrum = wr.OchiHubble(freq, freq_hz=True)

    # Swell component (i.e., j=1)
    hs_swell = 3.5
    tp_swell = 10.0
    q_swell = 2.0
    freq, vals_swell = spectrum(hs_swell, tp_swell, q=q_swell)

    # Wind component (i.e., j=2)
    hs_wind = 1.5
    tp_wind = 5.0
    q_wind = 2.0
    freq, vals_wind = spectrum(hs_wind, tp_wind, q=q_wind)

    # Total wave
    vals_tot = vals_swell + vals_wind

.. note::

    For the special case where :math:`q = 1`, Ochi-Hubble corresponds to the modified Pierson-Moskowitz
    spectrum.


Torsethaugen
------------
The *Torsethaugen* spectrum allows you to set up a double-peaked spectrum that represents
sea states which includes both a remotely generated swell component (with low frequency energy)
and a locally generated wind component (with high frequency energy). The spectral model
was developed based on average measured spectra for Norwegian waters (Haltenbanken and Statfjord).
The Torsethaugen spectrum is described by two parameter (i.e. :math:`H_s` and :math:`T_p`), and
is given by:

.. math::

    S(\omega) = \alpha_{\gamma}S_1(\omega; H_{s1}, T_{p1})\gamma^{exp\left( -\frac{(\omega - \omega_{p1})^2}{2\sigma^2\omega_{p1}^2} \right)} + S_2(\omega; H_{s2}, T_{p2})

where,

.. math::

    S_i(\omega) = \frac{A_{i}}{\omega^4} \exp(-\frac{B_{i}}{\omega^4})

Furthermore,

- :math:`A_i = \frac{3.26}{16} H_{si}^2 \omega_{pi}^3`
- :math:`B_i = \omega_{pi}^4`
- :math:`H_{si}` is the significant wave height for wave component :math:`i`.
- :math:`\omega_{pi} = \frac{2\pi}{T_{pi}}` is the angular spectral peak frequency for wave component :math:`i`.
- :math:`\gamma` is a peak enhancement factor.
- :math:`\alpha_{\gamma}` is a normalizing factor.
- :math:`\sigma` is the spectral width parameter, given by:

.. math::
    \sigma =
    \begin{cases}
        \sigma_a & \quad \text{if } \omega \leq \omega_{p1}\\
        \sigma_b & \quad \text{if } \omega > \omega_{p1}
    \end{cases}

.. note::

    The Torsethaugen spectrum is implemented according to the `original paper <https://www.sintef.no/globalassets/upload/fiskeri_og_havbruk/havbruksteknologi/2004-jsc-193.pdf/>`_
    by Torsethaugen and Haver published in 2004. Refer to this paper for full implementation
    details.

The :class:`~waveresponse.Torsethaugen` class provides functionality for generating a 1-D
Torsethaugen spectrum given two parameters (i.e., :math:`H_s`, :math:`T_p`):

.. code:: python

    import numpy as np
    import waveresponse as wr


    freq = np.arange(0.01, 1, 0.01)
    spectrum = wr.Torsethaugen(freq, freq_hz=True)

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
    S(\omega) = \int_0^{2\pi} S(\omega, \theta) d\theta

we get the following requirement for the spreading function for each frequency,
:math:`\omega_i`:

.. math::
    \int_0^{2\pi} D(\omega_i, \theta) d\theta = 1

In general, the spreading function is a function of both frequency, :math:`\omega`,
and direction, :math:`\theta`. However, it is common to use the same spreading
for all frequencies.

With ``waveresponse`` it is easy to construct a (directional) :class:`~waveresponse.WaveSpectrum`
object from a 1-D frequency spectrum and a spreading function:

.. code:: python

    import numpy as np
    import waveresponse as wr


    freq = np.arange(0.01, 1, 0.01)
    dirs = np.linspace(0.0, 360.0, endpoint=False)
    hs = 3.5
    tp = 10.0
    dirp = 45.0

    _, spectrum1d = wr.JONSWAP(freq, freq_hz=True)(hs, tp)
    spread_fun = wr.CosineFullSpreading(s=2, degrees=True)

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

A multimodal wave spectrum (with more than one peak) can be constructed by adding
together two (or more) wave spectrum components. E.g., if you have one swell and
one wind spectrum component, you can construct a two-peaked directional wave spectrum by:

.. math::

    S_{tot}(\omega, \theta) = S_{swell}(\omega, \theta) + S_{wind}(\omega, \theta)

This can be done by adding together two different :class:`~waveresponse.WaveSpectrum` objects:

.. code:: python

    wave_tot = swell + wind


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

where :math:`s` is a spreading coefficient, and :math:`\Gamma` is the Gamma function.


The :class:`~waveresponse.CosineHalfSpreading` class provides directional spreading
according to:

.. math::

    \kappa(\hat{\theta}) =
    \begin{cases}
        \frac{2^{2s}}{\pi} \frac{\Gamma^2(s+1)}{\Gamma^2(2s+1)} cos^{2s} (\hat{\theta}) & \quad \text{if } -\frac{\pi}{2} \leq \hat{\theta} \leq \frac{\pi}{2}\\
        0 & \quad \text{otherwise}
    \end{cases}


where :math:`s` is a spreading coefficient, and :math:`\Gamma` is the Gamma function.

In addition, the spreading functions in ``waveresponse`` cand determine discrete
direction bins with equal energy:

.. code:: python

    import waveresponse as wr


    spread_fun = wr.CosineFullSpreading(s=2, degrees=True)
    discrete_dirs = spread_fun.discrete_directions(5, direction_offset=0.0)


which may be used to spread 1-D 'non-directional' wave spectrum into waves
with equal energy.