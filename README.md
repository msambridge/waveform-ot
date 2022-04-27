# waveform-ot

This package provides a pure-Python implementation of algorithms described in [Sambridge, Jackson & Valentine, Geophysical Journal International, 2022](https://academic.oup.com/gji/advance-article-abstract/doi/10.1093/gji/ggac151/6572363).

The purpose is to demonstrate the theory described in the above paper while also providing examples of access to the underliying Optimal Transport library `OTlib.py` and time series Fingerprint library `FingerprintLib.py`.

## Contents
- [Introduction](#introduction)
- [Jupyter notebooks](#Jupyter-notebooks)
    - [Seismogram Software pyprop8](#Seismogram-software-pyprop8)
- [Citing this package](#citing-this-package)
- [Acknowledgements](#acknowledgements)

## Introduction
The current package demonstrates the following calculations:

- Computation of time series *Fingerprints* as a 2D density field representation of a time series suitable for use with Optimal Transport.
- Computation of 1D time and amplitude marginals from the 2D density field.
- Calculation of Wasserstein distances between observed and predicted  time and amplitude marginals using formulae in Sambridge et al. (2022)
- Calculation of derivatives of Wasserstein distances with respect to time series amplitudes.
- Demonstration of optimisation of Wasserstein misfits to fit noisy Double Ricker wavelets as a function of three parameters.
- Demonstration of optimisation of Wasserstein misfits to fit noisy seismic displacement waveforms (produced by package [pyprop8](https://github.com/valentineap/pyprop8)) based on a 1D seismic velocity model.


## Jupyter notebooks
To access core functionality, import `OTlib` and `FingerprintLib` within your Python code. This
provides access to the following:

- A class for specifying a seismic Fingerprint, `waveformFP`;
- A class for specifying a 1D of 2D density function suitable for use with OT calculation library, `OTpdf`;
- Several routines to operate on these classes to calculate Wasserstein distance between 1D or 2D PDFs as well as calculate derivatives with respect to the amplitude of 1D or 2D densities, and in turn to time series amplitudes. 

Calculations in the paper are demonstrated through a series of Jupyter notebooks demonstrating:
- Calculation of Wasserstein distances between PDFs represented as a finite sum of point masses in 1D.
- Fitting of a noisy double Ricker wavelet by minimization of L2 and Wasserstein misfits.
- Fitting of 33 noisy displacement seismograms for seismic source parameters by minimization of L2 and Wasserstein misfits.
- Calculation of derivatves of Wasserstein distances with respect to time series amplitudes.

Each jupyter notebook may be found in a separate directory and all rely on the library `OTlib.py`.

#### Seismogram Software pyprop8

This notebook makes use of Andrew Valentine's pyprop8 implementation of of the seismogram calculation algorithm set out in [O'Toole & Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x), together with the source derivatives set out in [O'Toole, Valentine & Woodhouse (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x). To run this notebook this package
needs to be installed separately. Instructions of how to do this appear [here](https://pypi.org/project/pyprop8/).

## Citing this package
If you make use of this code, please acknowledge the work that went into developing it! In particular, if you are preparing a publication, we would appreciate it if you cite the paper describing the general method used here:

- [Sambridge, M., Jackson, A., & Valentine, A. P. (2022)](https://academic.oup.com/gji/advance-article-abstract/doi/10.1093/gji/ggac151/6572363) "Geophysical Inversion and Optimal Transport", Geophysical Journal International, *in press*.


An appropriate form of words might be, "We make use of the software package `OTlib.py` (REF tbc), which is based on the approach of Sambridge, M., Jackson, A., & Valentine, A. P. (2022)."

## Acknowledgements

This package was developed at the Australian National University by Malcolm Sambridge & Andrew Valentine. 

This work has received support from the Australian Research Council under grants DE180100040 and DP200100053 and The Commonwealth Scientific Industrial Research Organisation Future Science Platform for Deep Earth Imaging.

