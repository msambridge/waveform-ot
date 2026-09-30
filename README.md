# waveform-ot

This package implements various algorithms described in [Sambridge, Jackson & Valentine, *Geophysical Journal International*, 2022](https://academic.oup.com/gji/advance-article-abstract/doi/10.1093/gji/ggac151/6572363). It is intended to serve as a demonstration of the theory described in the above paper, while also providing examples of access to the underlying Optimal Transport library `OTlib.py` and time series Fingerprint library `FingerprintLib.py`.

## Contents
- [Introduction](#introduction)
- [Package contents](#Package-contents)
- [Installation and usage](#installation-and-usage)
    - [Precomputed results (`pickles/`)](#precomputed-results-pickles)
    - [Seismogram software `pyprop8`](#Seismogram-software-pyprop8)
    - [Versions](#versions)
- [Citing this package](#citing-this-package)
- [Acknowledgements](#acknowledgements)

## Introduction
The current package demonstrates python software to calculate Wasserstein distances between 1D oscillatory time series, as well as the derivatives of those misfit measures with respect to the amplitude and origin time of the time series.

## Package contents

The core computational routines of this package are found in two Python modules, both found in the `libs` directory:

- `FingerprintLib` - Routines to calculate time series fingerprints, including a class `waveformFP`;
- `OTlib`  - Routines to undertake Optimal Transport calculations between fingerprints, including a class `OTpdf`.

The `libs` directory contains several additional modules: these provide various functions that are specific to the examples presented here.

Calculations in the paper are demonstrated through a sequence of Jupyter notebooks:
- `Point_mass_demo_Fig_5.ipynb`: An illustration of use of the OT library to calculate Wasserstein misfit between two 1D density functions. Also shows the intermediate steps in calculations of the Wasserstein distance and reproduces Figure 5 of the paper.
- `Ricker_Figs_1_7.ipynb`: Demonstrates calculation of a time series Fingerprint using a double Ricker wavelet example. Calculates Wasserstein misfit between noisy and noiseless waveforms as a function of the time shift and amplitude scale factor of the Ricker wavelets. Reproduces Figures 1 and 7 of the paper.
- `Ricker_Figs_3_8.ipynb`: Demonstrates calculation of 1D time and amplitude marginals from 2D Fingerprint density field. Shows how to use Wasserstein misfit for optimisation of 3 parameters controlling Ricker wavelets. Reproduces Figures 3 and 8 of the paper. 
- `Ricker_waveform_derivatives.ipynb`: Illustrates detailed calculation of derivatives of Wasserstein time and amplitude marginals with respect to waveform amplitudes and window origin time. Demonstrates application of chain rule to various intermediate derivatives as described in the paper. All calculations are generic except the final combination involving the forward problem model parameters. This notebook may be helpful in showing how to apply the libraries to other applications involving the fitting of time series.
- `source_location_cmt_W2L2_Figs_9_10_11.ipynb`: Demonstrates application of Wasserstein library to earthquake source and moment tensor inversion using example in the paper. Reproduces Figures 9, 10 and 11.
- `source_location_cmt_W2L2_Fig_12.ipynb`: Performs repeat inversions for earthquake source parameters starting from different source locations. Reproduces Figure 12 of the paper.

## Installation and usage

To explore the examples:
1. Ensure that you have a working Jupyter installation with a Python 3 kernel: see [Project Jupyter](https://jupyter.org/) for more details.
1. Obtain a copy of this package:
   - `git clone https://github.com/msambridge/waveform-ot.git`, or
   - Click the green 'Code' button on [this page](https://github.com/msambridge/waveform-ot/), select `Download ZIP`, and unzip the resulting file in an appropriate place on your system.
2. Ensure that the following modules are available on your system: `numpy`, `scipy`, `scikit-learn`, `tqdm`, `pyprop8`, `matplotlib`. A `requirements.txt` file is provided, so users of `pip` can simply `pip install -r requirements.txt`.
3. For the three notebooks that use precomputed results, download them into `pickles/` (see below).
4. Launch a Jupyter server, navigate to your `waveform-ot` directory, and run one or more of the notebooks.

### Precomputed results (`pickles/`)

Some calculations behind the figures are slow, so three notebooks can either compute their results or read them from files in the `pickles/` directory. That directory is empty in a fresh copy of the package. With their default settings these notebooks read the precomputed files, so download them first. The files used for the paper are attached to the [v1.0.2 release](https://github.com/msambridge/waveform-ot/releases/tag/v1.0.2) (about 134 MB). With the [GitHub CLI](https://cli.github.com/), from the `waveform-ot` directory:

```bash
gh release download v1.0.2 --dir pickles --pattern '*.pickle'
```

or download them from the release page and move them into `pickles/`. The release also includes `SHA256SUMS.txt` for checking the downloads.

| Notebook | Files | Contents |
|---|---|---|
| `Ricker_Figs_1_7.ipynb` | `Wasserstein_L2W2_misfits_ricker_low_res.pickle`, `Wasserstein_L2W2_misfits_ricker_high_res.pickle` | Misfit profiles and surfaces (L2, W1, W2) for the double Ricker wavelet |
| `source_location_cmt_W2L2_Figs_9_10_11.ipynb` | `OT_loc_low_res.pickle`, `OT_loc_high_res.pickle`, `L2_loc_low_res.pickle`, `L2_loc_high_res.pickle`, `L2_cmt.pickle` | Misfit grids over source location |
| `source_location_cmt_W2L2_Fig_12.ipynb` | `OT_solutions_loc.pickle`, `OT_solutions_cmt.pickle`, `L2_solutions_loc.pickle`, `L2_solutions_cmt.pickle` | Results of repeated inversions from many starting points |

Each of these notebooks also has settings near the top of its calculation section to compute the results itself and write its own pickle files instead; this can take from minutes to hours. The other notebooks don't use `pickles/`.

### Seismogram software `pyprop8`

This notebook makes use of Andrew Valentine's `pyprop8` implementation of the seismogram calculation algorithm set out in [O'Toole & Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x), together with the source derivatives set out in [O'Toole, Valentine & Woodhouse (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x). Detailed installation and usage instructions can be found [here](https://pyprop8.readthedocs.io/); `pip install pyprop8` may suffice for many users.

### Versions

The version used for the paper is tagged [`v1.0.2`](https://github.com/msambridge/waveform-ot/releases/tag/v1.0.2). Later versions of `OTlib.py` (`OTlib.__version__`) add further routines and fix bugs while staying compatible with these examples, which give identical results with them. Tests for `OTlib.py` are in `tests/`: run `python tests/test_otlib_wasser.py` from the `waveform-ot` directory.

## Citing this package
If you make use of this code, please acknowledge the work that went into developing it. In particular, if you are preparing a publication, we would appreciate it if you cite the paper describing the general method used here:

- [Sambridge, M., Jackson, A., & Valentine, A. P. (2022)](https://academic.oup.com/gji/advance-article-abstract/doi/10.1093/gji/ggac151/6572363) "Geophysical Inversion and Optimal Transport", Geophysical Journal International, ggac151, doi:10.1093/gji/ggac151.


An appropriate form of words might be, "We make use of the software package `OTlib.py`, which implements the approach of Sambridge *et al.* (2022)."

## Acknowledgements

This package was developed at the Australian National University by Malcolm Sambridge & Andrew Valentine. 

This work has received support from the Australian Research Council under grants DE180100040 and DP200100053 and The Commonwealth Scientific Industrial Research Organisation Future Science Platform for Deep Earth Imaging.

