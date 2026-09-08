# Part B Renderer Implementation Plan

This document outlines the architecture and tasks for Part B of the renderer: Reflectance, Texturing, Sensor Modelling, and Domain Randomization.

## User Review Required
> [!IMPORTANT]
> **View Vector Definition:** The Lunar-Lambertian model requires a view vector ($v$). For most orbital imaging, nadir pointing ($v = [0, 0, 1]$) is a safe assumption. If we need to support off-nadir images, should we parse the `roll`/`pitch` fields from the source telemetry, or will assuming a nadir-pointing camera suffice for the initial ML training dataset?

> [!IMPORTANT]
> **Albedo/Texture Maps:** Constant albedo was used for validation. Do we have access to a registered albedo mosaic (like LROC WAC), or should we procedurally generate albedo variation (e.g., Perlin noise layered over the DEM to simulate mare/highland transitions) for domain randomization?

## Proposed Architecture

To fulfill the requirements of §6.3, §6.5–6.7 while keeping the codebase decoupled, we will introduce the following modules:

### 1. `reflectance.py`
Handles the physics of how light bounces off the lunar regolith.
- **Lunar-Lambertian Model**: Will combine a Lommel-Seeliger term (for backscatter) and a Lambertian term.
  - Formula: $I = A \times \left[ L(\alpha) \frac{\mu_0}{\mu_0 + \mu} + (1 - L(\alpha)) \mu_0 \right]$
  - Inputs: Surface normal ($n$), Sun vector ($s$), View vector ($v$).
- **Albedo Draping**: A function to overlay spatially varying albedo maps ($A$) onto the DEM grid.

### 2. `sensor.py`
Simulates the imperfections and physics of the optical system and CCD.
- **PSF Blur**: Gaussian/Airy disk convolution to simulate the Point Spread Function of the camera lens.
- **Noise Injection**:
  - **Shot Noise**: Poisson noise applied to the linear radiance.
  - **Read Noise**: Gaussian noise added to simulate thermal/electronic sensor noise.
- **Quantization**: Mapping the continuous linear signal into an 8-bit, 10-bit, or 12-bit digital number (DN) space, replicating the A/D converter of the target camera (OHRC/NAC).

### 3. `domain_randomization.py`
A pipeline orchestrator designed to generate large volumes of training data for ML algorithms.
- Takes the fixed geometry buffers from Part A (Normals + Horizon Maps) and loops through randomization parameters:
  - **Lighting**: Randomly sample sun azimuth $[0, 360]$ and sun elevation $[5^\circ, 85^\circ]$.
  - **Albedo**: Randomly shift the mean albedo or procedural noise frequency.
  - **Sensor characteristics**: Randomize the PSF blur radius ($\sigma \in [0.5, 2.0]$) and signal-to-noise ratio.

## Verification Plan
### Automated Tests
- Build a unit test for `reflectance.py` verifying that maximum radiance occurs at phase angle $0$ (opposition surge approximation).
- Build a unit test for `sensor.py` to assert that the noise variance statistically matches the injected Poisson/Gaussian parameters.

### Manual Verification
- Render a single DEM tile through the entire Part B pipeline.
- Visually compare the generated output side-by-side with a real cropped OHRC/NAC image of the exact same footprint to qualitative evaluate the photorealism (shadow contrast, noise floor, and optical softness).
