# PIVPy Development Roadmap

For full architectural designs and implementation details, see:
- [PIVMat & PyPostPiv Parity Roadmap](docs/architecture/pivmat-parity-roadmap.md)
- [Zarr-First Storage Architecture & API Design](docs/architecture/zarr-migration.md)

---

## Executive Summary

PIVPy's development roadmap synthesizes key capabilities from MATLAB's **PIVMat 4.22** toolbox (F. Moisy) and the Python **PyPostPiv** library (J. Hu et al., Univ. of Waterloo), delivering a modern, high-performance, out-of-core Pythonic experience:

1. **Phase 1: Synthetic Flow Generators & Benchmarking Suite**
   - Analytical vortices (Burgers, Lamb-Oseen, Rankine, Vatistas)
   - Multi-vortex synthetic turbulence fields
   - Random divergence-free vector fields
   - Synthetic speckle pattern generation

2. **Phase 2: Vortex Identification Criteria & Flow Topology**
   - Normalized Angular Momentum ($\Gamma_1$ & $\Gamma_2$)
   - Circulation-based noise-robust vorticity
   - Solid-body rotation subtraction (`subsbr`)
   - Invariant metrics ($Q$-criterion, Okubo-Weiss parameter)

3. **Phase 3: Spatial Filtering, Geometry Masking & Gradient Calculus**
   - High-order and robust gradient schemes (`2nd_central`, `4th_central`, `least_squares`)
   - 2D spatial median filtering for outlier removal
   - Circular, rectangular, and arbitrary polygon obstacle masking
   - Arbitrary grid remapping (`remapf`), rotation, and flipping

4. **Phase 4: Temporal Frequency Analysis & Modal Filtering**
   - Fourier temporal bandpass and notch filtering (`tempfilterf`)
   - Unsteady shedding mode isolation and phase-averaging

5. **Phase 5: Turbulence Statistics, TKE & Structure Functions**
   - Velocity fluctuations ($u', v', w'$), RMS, and Turbulent Kinetic Energy ($k$)
   - Multi-order longitudinal and transverse velocity structure functions ($S_p(r)$)
   - Full Reynolds stress tensor and Lumley anisotropy invariants
   - 2D/1D spatial wavenumber energy spectra ($E(k)$)

6. **Phase 6: Multi-Camera, Stereoscopic 3C & Ingestion Pipeline**
   - Stereoscopic 3-component ($u, v, w$) pipeline support across all accessors
   - Direct batch DaVis `.vc7`/`.set` to Zarr conversion tool (`convert_vc7_to_zarr`)

