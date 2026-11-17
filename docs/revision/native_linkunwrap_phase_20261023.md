# Native-grid within-link unwrap phase control — 2026-10-23

**Completed and verified: 9,000 rows.** Variant: `native_time_wrapped_linear_link_unwrap_v1`.

Dataset directory: `D:\pa-csi-dg-revision-runtime\data\native_linkunwrap_phase_v1`. It contains three float32 `[3000,850,90]` phase arrays, six source/target manifests, three subject sidecars, complete record identities and hashes, frozen code, and synthetic/input QA. Native C amplitude and labels are referenced without duplication or modification.

## Frozen definition

- **purpose:** Within-receive-link unwrap control of the historical-style phase on the native-time linear amplitude grid; not physical calibration or repaired phase
- **input:** Permutation-corrected native DAT complex128 CSI [packet, 90], 3 Rx groups x 30 columns
- **timestamp:** Native uint32 microseconds, modulo-2^32 forward deltas, reject deltas >=2^31, start at zero
- **phase_extraction:** np.angle(csi) float64, before interpolation; angle(0+0j)=0 retained as the historical arithmetic convention
- **duplicates:** Arithmetic mean of wrapped angle values at identical elapsed times, independently per column; not a circular mean
- **grid:** np.linspace(0.0, native_elapsed[-1], 850), float64, endpoints included; identical to frozen native_time_linear amplitude
- **resampling:** scipy.interpolate.interp1d(kind=linear, axis=0, bounds_error=True) on wrapped angle scalars using the frozen uniform_interpolation helper
- **large_gaps:** Linearly bridge all packet gaps without a mask or cutoff; no row exclusion
- **feature_unwrap:** After interpolation apply np.unwrap(..., axis=-1) in float64 with defaults (period 2*pi, discont pi) separately to columns 0-29, 30-59 and 60-89; the 29->30 and 59->60 receive-link boundaries are NOT crossed
- **only_difference_from_reference:** native_time_wrapped_linear_feature_unwrap_v1 unwraps all 90 columns as one sequence; everything else is identical
- **adjacent_frames:** No temporal unwrap, temporal centering, phase differencing, filtering, or train-fitted normalization
- **cast:** Cast only the final [850,90] phase matrix to float32; loader must not unwrap again
- **amplitude:** Reference existing native_time_linear amplitude files and labels without copying or modifying them
- **model_view_slicing:** The model pseudo-Tx convention (mode=tx,num_rx=1,num_tx=3,num_subcarriers=30) slices the same three 30-column software feature groups; it does not assert three physical transmitters

## Verification and interpretation

- Every DAT byte stream was checked against SHA-256, pinned Git blob and the native preparation identity; original DAT-to-MAT mappings, labels and partition order are retained.
- All 9,000 output rows passed shape, float32, finite and per-row content-hash verification. Six manifests preserve C sample identities and source subject/acquisition separation.
- Eight input-selected records passed independent per-column interpolation and explicit per-link unwrap replay (maximum error 0, tolerance 1e-4); their C amplitude reconstruction is exactly equal to the frozen arrays.
- Against the frozen cross-link array (`native_time_wrapped_linear_feature_unwrap_v1`), receive link 0 is bitwise equal in all rows, and links 1 and 2 differ only by one whole-turn offset per row, frame and link (maximum residual 1.03e-05 rad). The cross-link unwrap shifted link 1 in 84.88% and link 2 in 87.55% of frames; 0 of 9,000 rows are identical in both chains.
- This is a within-link unwrap control on the common native grid. It is not a physically calibrated phase. The only difference from the historical-style chain is whether the feature-axis unwrap crosses the two receive-link boundaries.
- No model was trained or scored during this operation. Training must use already processed phase directly and must not apply another unwrap.

Dataset manifest SHA-256: `6a78a4ee0aa0ad598b40d04bd65c22e44759de6a344683f8e483f39be0117f38`.
