# Native-grid historical-style phase diagnostic — 2026-09-29

**Completed and verified: 9,000 rows.** Variant: `native_time_wrapped_linear_feature_unwrap_v1`.

Dataset directory: `D:\pa-csi-dg-revision-runtime\data\native_histstyle_phase_v1`. It contains three float32 `[3000,850,90]` phase arrays, six source/target manifests, three subject sidecars, complete record identities and hashes, frozen code, and synthetic/input QA. Native C amplitude and labels are referenced without duplication or modification.

## Frozen definition

- **purpose:** Historical-style diagnostic phase representation on the native-time linear amplitude grid; not physical calibration or repaired phase
- **input:** Permutation-corrected native DAT complex128 CSI [packet, 90], 3 Rx groups x 30 columns
- **timestamp:** Native uint32 microseconds, modulo-2^32 forward deltas, reject deltas >=2^31, start at zero
- **phase_extraction:** np.angle(csi) float64, before interpolation; angle(0+0j)=0 retained as the historical arithmetic convention
- **duplicates:** Arithmetic mean of wrapped angle values at identical elapsed times, independently per column; not a circular mean
- **grid:** np.linspace(0.0, native_elapsed[-1], 850), float64, endpoints included; identical to frozen native_time_linear amplitude
- **resampling:** scipy.interpolate.interp1d(kind=linear, axis=0, bounds_error=True) on wrapped angle scalars using the frozen uniform_interpolation helper
- **large_gaps:** Linearly bridge all packet gaps without a mask or cutoff; no row exclusion
- **feature_unwrap:** After interpolation apply np.unwrap(..., axis=-1) in float64 using defaults (period 2*pi, discont pi); columns 29->30 and 59->60 are crossed
- **adjacent_frames:** No temporal unwrap, temporal centering, phase differencing, filtering, or train-fitted normalization
- **cast:** Cast only the final [850,90] phase matrix to float32; loader must not unwrap again
- **amplitude:** Reference existing native_time_linear amplitude files and labels without copying or modifying them
- **interpretation:** Whole phase-processing-chain comparison against existing differential phase, not isolation of the differential operation alone

## Verification and interpretation

- Every DAT byte stream was checked against SHA-256, pinned Git blob and the native preparation identity; original DAT-to-MAT mappings, labels and partition order are retained.
- All 9,000 output rows passed shape, float32, finite and per-row content-hash verification. Six manifests preserve C sample identities and source subject/acquisition separation.
- Eight input-selected records passed independent per-column interpolation and explicit unwrap-formula replay (maximum error 0, tolerance 1e-4); their C amplitude reconstruction is exactly equal to the frozen arrays.
- Synthetic +/-3.1 rad observations interpolate through approximately zero (frame 424: 0.00365135), demonstrating known non-circular arithmetic. Opposing duplicate angles average to zero, angle(0)=0 is retained, and cross-Rx feature unwrap is explicitly checked.
- Retained 71161 zero-complex observations, 0 duplicate time steps, and all packet gaps, with maximum gap 2.015493 s. No rows were removed.
- This is a historical-style diagnostic on a common grid. It is not the old historical array, a physically calibrated phase, or evidence of improved recognition. The difference from the existing differential phase includes the entire phase-processing chain.
- No model was trained or scored during this operation. Training must use already processed phase directly and must not apply another unwrap.

Dataset manifest SHA-256: `812e3394f06f21abf04d3ef2366c92ab2eeca7afb2478419da60671820da6c67`.
