# PIVPy Development Roadmap

For full architectural designs and implementation details, see:
- [PIVMat Parity & Modernization Roadmap](docs/architecture/pivmat-parity-roadmap.md)
- [Zarr-First Storage Architecture & API Design](docs/architecture/zarr-migration.md)

---

## Executive Summary

PIVPy's development roadmap is divided into thematic phases aimed at achieving feature parity with MATLAB's PIVMat 4.22 toolbox while delivering a modern, high-performance, out-of-core Pythonic experience:

1. **Phase 1: Synthetic Flow Generators & Benchmarking Suite**
   - Analytical vortices (Burgers, Lamb-Oseen, Rankine, Vatistas)
   - Multi-vortex synthetic turbulence fields
   - Random divergence-free vector fields
   - Synthetic speckle pattern generation

2. **Phase 2: Vortex Identification Criteria & Flow Topology**
   - Normalized Angular Momentum ($\Gamma_1$ & $\Gamma_2$)
   - Solid-body rotation subtraction (`subsbr`)
   - Invariant metrics ($Q$-criterion, Okubo-Weiss parameter)

3. **Phase 3: Spatial Filtering, Geometry Masking & Grid Transforms**
   - 2D spatial median filtering for outlier removal
   - Circular, rectangular, and arbitrary polygon obstacle masking
   - Arbitrary grid remapping (`remapf`), rotation, and flipping

4. **Phase 4: Temporal Frequency Analysis & Modal Filtering**
   - Fourier temporal bandpass and notch filtering (`tempfilterf`)
   - Unsteady shedding mode isolation and phase-averaging

5. **Phase 5: Turbulence Statistics, Spectra & Structure Functions**
   - Multi-order longitudinal and transverse velocity structure functions ($S_p(r)$)
   - Full Reynolds stress tensor and Lumley anisotropy invariants
   - 2D/1D spatial wavenumber energy spectra ($E(k)$)
