# Instrument Data Simulation and Observation Error Propagation for GRACE-FO 

## Summary


Satellite gravimetry provides a unique means of observing mass redistribution within the Earth system, with applications in hydrology, cryospheric science, oceanography, solid Earth geophysics, and geodesy. Missions such as GRACE and GRACE-FO infer gravity-field variations from precise intersatellite tracking between two low-Earth-orbiting spacecraft. The quality of the derived gravimetric information is limited by measurement errors affecting the observation chain. This work investigates how different instrument observation-error models propagate into derived intersatellite observables and line-of-sight gravity-difference errors for a satellite gravimetry mission in a nearly circular trailing formation.

A simulation framework is implemented in Tudat to propagate the two-spacecraft dynamics, incorporating a spherical-harmonic representation of Earth's gravity field, third-body gravitational perturbations, relativistic corrections, atmospheric drag, and solar radiation pressure. The spacecraft is represented by a simplified GRACE-FO macromodel, whose optical and aerodynamic properties are tuned to reproduce representative non-gravitational acceleration behaviour. The relative geometry is initialized from GRACE-FO TLE data, and an impulsive phasing strategy is used to maintain the intersatellite separation within a prescribed deadband.

Synthetic Level-1B-like data products are generated for GPS navigation, attitude, accelerometer, and K-band ranging observations. Two stochastic-noise variants are considered. Variant 1 represents random errors using simplified white Gaussian noise assumptions, whereas Variant 2 introduces frequency-dependent stochastic realisations based on prescribed spectral-density models. The resulting error histories are propagated into intersatellite range, range-rate, range-acceleration, and line-of-sight gravity-difference errors. Spectral estimates are obtained using Welch's method, and time-domain finite-difference derivatives are compared with frequency-domain analytical differentiation.

The results show that the selected observation-error model affects the simulated measurement chain in a frequency-dependent and observable-specific manner. At the range level, Variant 1 produces an approximately flat error amplitude spectral density, while Variant 2 introduces a pronounced low-frequency coloured component. After propagation to range rate and range acceleration, the spectra are reshaped by differentiation, which amplifies high-frequency content. Numerical and analytical differentiation agree well over the low- and intermediate-frequency bands, while systematic discrepancies appear close to the Nyquist frequency due to the discrete transfer function of the finite-difference operators.

At the line-of-sight gravity-difference level, the error spectrum is not a direct effect of the K-band range-error spectrum. Instead, it results from the combined influence of range derivatives, GPS-derived relative motion, accelerometer observations, attitude-dependent corrections, and line-of-sight projection. Although the two stochastic-noise variants differ most strongly at low frequencies in the range observable, their differences become more relevant at high frequencies in the propagated gravity-difference error.

Overall, simplified white-noise assumptions are insufficient to fully characterize observation-error propagation in satellite gravimetry simulations. A realistic assessment requires modelling both the magnitude and spectral structure of each instrument error source, as well as its transformation through the complete measurement-processing chain. Future work should extend the framework toward complete gravity-field recovery, improved non-gravitational force modelling, more realistic instrument-error models, Laser Ranging Interferometer observations, alternative differentiation strategies, and validation against real GRACE-FO Level-1B data.

## Contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Required input files ](#required-input-files)
- [Running the simulation](#running-the-simulation)
- [Configuration](#configuration)
- [Propagation cache](#propagation-cache)
- [Outputs](#outputs)
- [Optional real-data downloader](#optional-real-data-downloader)

## Repository structure

First, create and activate the project environment using the root-level [`environment.yml`](environment.yml), following the installation instructions below. This file installs the prebuilt Tudat package and the scientific dependencies required by the analysis. Run all subsequent commands from the repository root so that the scripts can resolve their relative input and output paths.

Before starting the simulation, verify that [`data/`](data/) contains the spacecraft geometry, Earth texture, and prescribed amplitude spectral density (ASD) curves, and that the additional SPICE kernels are present in `kernels/`. Instructions for obtaining missing kernels are provided below.

Next, run `python src/main.py` and select either noise variant `1` or `2` when prompted. [`src/main.py`](src/main.py) is the primary entry point: it constructs the simulation environment, propagates the spacecraft or loads matching cached propagation results, generates synthetic observations, and analyzes their errors. After this run finishes, execute the same command again and select the variant which was not initially selected. Keep the physical and numerical settings unchanged between runs so that the comparison isolates the effect of the noise model. The variants may also be run in the opposite order.

Once both variants have produced their spectra, run `python src/noise_version_comparison.py`. This script reads the saved debiased range-error and LGD-error spectra and plots their absolute ASD differences. If both sets of compatible spectra already exist, this comparison can be run directly without repeating the analyses performed by `python src/main.py`. The results are stored under `output/`.

Two additional scripts are optional and independent of this sequence. [`src/finite_difference_method_test.py`](src/finite_difference_method_test.py), run with `python src/finite_difference_method_test.py`, validates numerical differentiation against simulation truth using simplified orbital models. It can be run before the full analysis, but its results are not required by `main.py`. [`src/data_downloader.py`](src/data_downloader.py), run with `python src/data_downloader.py` downloads and extracts selected real GRACE-FO Level-1B products. Those downloaded products are not currently consumed by the synthetic simulation, so downloading them is not a prerequisite.

## Installation

Use the root-level [`environment.yml`](environment.yml) to install all required analysis packages together. The supported installation targets are **Linux** and **Windows with a Linux distribution under WSL2**. Both use the same environment file and prebuilt Tudat package. The pinned Tudat release has Windows binaries, but the complete project also requires PyCBC. The [PyCBC 2.10.0 release files](https://pypi.org/project/PyCBC/2.10.0/#files) provide Linux and macOS wheels, not Windows wheels. Therefore, native Windows is not a verified or supported installation route for this environment.

### Linux setup

On Ubuntu or Debian, install the command-line tools used to obtain the repository and kernels if they are not already available:

```bash
sudo apt update
sudo apt install --yes git curl ca-certificates
```

If Conda is already installed, skip the following installer steps and check it with `conda --version`. Otherwise, install the Linux edition of Miniconda using the [official instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html):

```bash
curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
```

Compare the printed SHA-256 digest with the matching filename in the [official installer index](https://repo.anaconda.com/miniconda/) before executing the installer. If it matches, run:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the installer prompts, choose a user-writable installation directory, and allow shell initialization. Reopen the terminal and run `conda --version`. Open a checkout of this repository, change to its root directory, and continue with [create the project environment](#create-the-project-environment).


### Windows setup: WSL2

Use WSL2 to run the Linux environment on Windows. The standard installation command requires running Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11. Open **PowerShell as Administrator**, run the following command, and restart Windows if requested. See [Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).

```powershell
wsl --install -d Ubuntu
```

Open Ubuntu from the Start menu and complete the Linux username/password setup. In PowerShell, check the distribution version:

```powershell
wsl --list --verbose
```

The Ubuntu entry should show version `2`. If an existing Ubuntu installation shows version `1`, convert it with `wsl --set-version Ubuntu 2`, substituting its actual distribution name if different.

**Run all remaining commands inside the Ubuntu terminal.** Follow the Linux setup above to install the **Linux** edition of Conda inside WSL, then create the project environment below. A Conda installation on Windows does not replace the separate Linux installation.

Keep the repository in the WSL Linux filesystem, for example `/home/<linux-user>/satellite_gravimetry`, for better Linux filesystem performance. A Windows path such as `C:\Users\<user>\satellite_gravimetry` is accessible as `/mnt/c/Users/<user>/satellite_gravimetry`, but is not the preferred location for this workflow (see [Microsoft filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)).

### Create the project environment

In a Linux terminal or the Ubuntu terminal under WSL2, change to the repository root, then create and activate the environment:

```bash
conda env create -f environment.yml
conda activate satellite-gravimetry
```

 If an environment named `satellite-gravimetry` already exists, choose a new name with `conda env create -n <new-name> -f environment.yml` and activate that name.

The file uses the development channel described in the [official Tudat installation guide](https://docs.tudat.space/en/latest/getting-started/installation.html#development-version), with Tudat explicitly pinned to `1.0.0.dev74`. 

The downloader is optional; its installation is described separately [below](#optional-real-data-downloader).

### Verify the active interpreter and required APIs

From the repository root, with `satellite-gravimetry` active, run these commands in the Linux/WSL terminal:

```bash
python -m pip check
```

The command should report `No broken requirements found.` 

These are installation smoke checks, not a complete numerical validation of the mission model. Configure the IDE to use the same interpreter printed by the script. 

## Required input files

Run commands from the repository root: most paths are relative to the current working directory.

### Project assets

The main workflow uses these files under `data/`:

| File | Role |
| --- | --- |
| `grace_fo_low_fidelity.dae` | Spacecraft panel geometry |
| `earth_texture.jpg` | Earth texture for visualization |
| `pitch_angles_asd_data.json` | Reference pitch-angle ASD |
| `yaw_angles_asd_data.json` | Reference yaw-angle ASD |
| `roll_angles_asd_data.json` | Reference roll-angle ASD |
| `relative_position_error_asd_data.json` | Relative-position error ASD used by GPS model |

The JSON curves' frequency coordinate is in Hz; attitude ASDs are in rad/√Hz and the relative-position ASD is in m/√Hz. 

### SPICE kernels

`main.py` first loads Tudat's standard kernels and then explicitly loads four additional files. Obtain missing files from the official NAIF archive and place them at exactly these paths:

| Expected local path | Official download |
| --- | --- |
| `kernels/nep105.bsp` | [Neptune-system SPK](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep105.bsp) |
| `kernels/plu060.bsp` | [Pluto-system SPK](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/plu060.bsp) |
| `kernels/ura182.bsp` | [Uranus-system SPK, archived version](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/ura182.bsp) |
| `kernels/gm_de440.pck` | [DE440 gravitational parameters](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc) |

NAIF publishes the last file with a `.tpc` extension; save it as `gm_de440.pck` to match the path in the code. 

For a new checkout without these files, run the following in the Linux terminal or the Ubuntu terminal under WSL2:

```bash
mkdir -p kernels
curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep105.bsp -o kernels/nep105.bsp
curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/plu060.bsp -o kernels/plu060.bsp
curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/ura182.bsp -o kernels/ura182.bsp
curl -fL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc -o kernels/gm_de440.pck
```

These download commands overwrite files with matching names; use them only for missing inputs or deliberate replacements. The main script does not download these kernels automatically. Replacing or copying kernels can change propagation-cache fingerprints.

## Running the simulation

Once dependencies and inputs are available, activate the environment and change to the repository root. For the recommended prebuilt setup:

```bash
conda activate satellite-gravimetry
cd satellite_gravimetry
```

Use the `satellite-gravimetry` environment created during installation (or the custom name selected when creating it).

1. Run the full analysis for variant 1:

   ```bash
   python src/main.py
   ```

   Enter `1` when prompted. A matching cache is reused; otherwise this run performs the orbital propagation. Wait for the complete post-processing stage to finish.

2. Run the same script for variant 2:

   ```bash
   python src/main.py
   ```

   Enter `2` when prompted. Keep the propagation configuration and input assets unchanged to share the cached orbit and make the noise comparison meaningful.

3. Compare the saved spectra:

   ```bash
   python src/noise_version_comparison.py
   ```

The two variants may be run in the opposite order; both must have produced the required spectra before comparison. No supporting module needs to be run manually before `main.py`.


If both variants' exported spectra already exist, the comparison can be run directly without a new propagation or kernel setup. Existing files may represent previous experiments: their presence alone does not establish matching simulation settings.

### Optional: standalone finite-difference validation

```bash
python src/finite_difference_method_test.py
```

This independent script propagates simplified spacecraft-pair cases under point-mass and J2-perturbed Earth gravity over one day, starting on 1 March 2005. It compares numerical acceleration and line-of-sight acceleration estimates with simulation truth for several finite-difference accuracy orders and prints coefficient information.

It uses Tudat's standard kernels, not the four extra kernels required by `main.py`. It is not a prerequisite for either noise variant and is not a pytest test suite. Its current figure directory is `GRACE-FO/plots/finite_difference_method_validation/`, which differs from the main workflow's `output/plots/` directory.

## Configuration

Settings are edited in Python; there is no separate runtime configuration file or general command-line configuration interface.

| Setting | Current default | Location |
| --- | --- | --- |
| Simulation interval | 1 January–21 February 2019: 51 days | `main.py`: `simulation_start_epoch`, `simulation_end_epoch` |
| Nominal integration/sample step | 5 s | `main.py`: `time_step` |
| Earth gravity truncation | Degree 200, order 200 | `main.py`: acceleration settings and cache configuration |
| Spacecraft mass | 655 kg each | `main.py`: `grace_fo_mass` |
| Formation distance threshold | 25 km departure from reference separation | `main.py`: `guidance_distance_threshold_m` |
| Phasing duration parameter | 1 revolution | `main.py`: `guidance_n_revolutions` |
| Post-phasing cooldown | 3,600 s | `main.py`: `guidance_cooldown_duration_s` |
| Spectral target frequency resolution | 5 × 10⁻⁶ Hz | `differentiator.py`: `LOWEST_FREQUENCY_RESOLUTION_HZ` |
| Welch sensitivity-search candidates | 10,000 per derivative search | `differentiator.py`: `sensitivity_analysis_runs` |

Important dependencies between these settings:

- **Changing `main.py`'s time step alone is insufficient.** Several methods in `noise_generator.py` explicitly set `time_step = 5.0`, including GPS noise, reference pointing angles, pointing-angle measurement noise, and K-band system/oscillator noise. Sampling, ASD normalization, interpolation, differentiation, and the cache configuration must remain consistent.
- **A very short run is not a drop-in full-pipeline test.** At the default resolution, Welch segments contain 40,000 samples at five-second spacing, or 200,000 s. The median-mean estimator needs at least two overlapping segments: 60,000 contiguous samples, approximately 3.47 days. Maneuver exclusions can leave no sufficiently long usable arc even when the total simulation spans several days. Short experiments require a deliberate adjustment of the spectral resolution and related analysis settings.
- **Dependent-variable column order is part of the internal interface.** Downstream routines access particular columns by index. Reordering or adding entries in `dependent_variables_to_save` requires reviewing those consumers and invalidating incompatible cached results.
- **Keep the propagation fingerprint synchronized with the model.** `propagation_config` manually summarizes the settings; it is not automatically derived from the complete Tudat setup or source code.


## Propagation cache

The propagation cache is shared between noise variants because the selected measurement-noise variant does not alter the reference orbital dynamics.

Each entry contains:

```text
output/cache/propagation_results/
  <configuration_sha256>.h5
  <configuration_sha256>_config.json
```

The HDF5 file stores state and dependent-variable histories, the guidance log, mean orbital period, and propagation timing/function-evaluation metadata. The JSON sidecar records the input configuration for inspection.

Cache behavior:

- A matching entry skips `propagate_orbits`; the console explicitly reports that cached propagation products were loaded.
- A cache miss runs the complete propagation and saves the results afterward.
- Reference-attitude generation, environment construction, some plotting, synthetic observations, differentiation, and spectral analysis still execute on a cache hit. Kernels and model assets are therefore still required when running `main.py` with a populated cache.
- The cache is not an intermediate checkpoint or restart mechanism. An interrupted first propagation cannot resume from a saved partial arc.
To force a fresh calculation while preserving old results, point `propagation_cache_dir` in `main.py` at a new empty directory. There is no dedicated force-recompute command-line flag.

The default setup is computationally substantial. The default 51-day scenario records approximately 88,548 seconds of propagation CPU time, about 24.6 hours.


## Outputs

| Location | Contents |
| --- | --- |
| `output/plots/version_1/` | Variant 1 orbit, guidance, attitude, observation-error, and spectra figures |
| `output/plots/version_2/` | Corresponding variant 2 figures |
| `output/spectra/version_1/` | Selected variant 1 spectra in HDF5 |
| `output/spectra/version_2/` | Selected variant 2 spectra in HDF5 |
| `output/plots/version_comparison/` | Absolute ASD-difference figures |
| `output/cache/propagation_results/` | Reusable propagation histories and configuration sidecars |
| `GRACE-FO/plots/finite_difference_method_validation/` | Figures from the optional standalone validation script |

Each variant's spectral directory contains:

| HDF5 filename | ASD units |
| --- | --- |
| `range_error_debiased_asd.h5` | m/√Hz |
| `range_rate_error_analytical_differentiated_asd.h5` | m s⁻¹/√Hz |
| `range_acceleration_error_analytical_differentiated_asd.h5` | m s⁻²/√Hz |
| `lgd_error_asd.h5` | m s⁻²/√Hz |

These files contain one-dimensional `frequencies` and `asd` datasets, with attributes identifying the spectrum, units, and noise variant. The range-rate and range-acceleration filenames explicitly identify the analytically differentiated spectra; they should not be mistaken for the full numerical derivative histories.

The comparison script creates `range_error_debiased_asd_absolute_difference.png` and `lgd_error_asd_absolute_difference.png`. 


## Optional real-data downloader

[`src/data_downloader.py`](src/data_downloader.py) is a separate utility wrapping the `podaac-data-downloader` executable supplied by `podaac-data-subscriber`. Downloaded observations are **not currently read by `main.py`**, so this is not an installation or execution prerequisite for the synthetic workflow.

Install the optional package in the active environment if needed:

```bash
python -m pip install podaac-data-subscriber
podaac-data-downloader --help
```

See the [official PO.DAAC package instructions](https://github.com/podaac/data-subscriber) and [downloader authentication documentation](https://github.com/podaac/data-subscriber/blob/main/Downloader.md). Configure an Earthdata Login account and the required local authentication before downloading. Store credentials outside this repository; if using `.netrc`, restrict access to the owning user and never commit it.

Review the example configuration at the bottom of the script, then run:

```bash
python src/data_downloader.py
```

The current example requests the `GRACEFO_L1B_ASCII_GRAV_JPL_RL04` collection for 1 January–1 February 2019, retaining `SCA1B` and `GNI1B` products under `data/grace_fo_l1b_ascii_grav_jpl_rl04_2019_01_01_2019_02_01/`.
