"""Read the stored MAT convention without inferring or repeating perm correction."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def integer_scalar(value, name, minimum, maximum):
    value = np.asarray(value)
    if value.size != 1 or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
        raise ValueError("Invalid numeric scalar " + name)
    scalar = value.item()
    if scalar != int(scalar) or scalar < minimum or scalar > maximum:
        raise ValueError("Non-integral or out-of-range scalar " + name)
    return int(scalar)


def relative_microseconds(timestamps):
    raw_values = np.asarray(timestamps)
    if raw_values.ndim != 1 or raw_values.size < 2:
        raise ValueError("At least two packet timestamps required")
    if (raw_values.dtype.kind not in "iuf" or not np.isfinite(raw_values).all()
        or np.any(raw_values != np.floor(raw_values))
        or np.any(raw_values < -(2**31)) or np.any(raw_values >= 2**32)):
        raise ValueError("Timestamp is not a signed/unsigned uint32 bit pattern")
    values = raw_values.astype(np.int64)
    bits = values % (2**32)
    deltas = (bits[1:] - bits[:-1]) % (2**32)
    if np.any(deltas >= 2**31):
        raise ValueError("Backward/ambiguous timestamp jump")
    elapsed = np.concatenate(([0], np.cumsum(deltas, dtype=np.int64)))
    if elapsed[-1] <= 0:
        raise ValueError("Non-positive record duration")
    return elapsed, {"wrap_count": int(np.sum(bits[1:] < bits[:-1])),
                     "duplicate_timestamp_count": int(np.sum(deltas == 0)),
                     "duration_microseconds": int(elapsed[-1]), "max_gap_microseconds": int(deltas.max())}


def read_stored_mat(path: Path):
    mat = loadmat(path)
    trace = mat.get("trace")
    if trace is None or trace.ndim != 2 or trace.shape[1] != 1:
        raise ValueError("Expected trace [packets,1] MATLAB cell array")
    csi, timestamps, perms = [], [], []
    for cell in trace[:, 0]:
        if cell.shape != (1, 1) or not cell.dtype.names:
            raise ValueError("Unexpected trace record structure")
        record = cell[0, 0]
        for name in ("timestamp_low", "Nrx", "Ntx", "perm", "csi"):
            if name not in record.dtype.names:
                raise ValueError("Missing trace field " + name)
        nrx, ntx = integer_scalar(record["Nrx"], "Nrx", 1, 3), integer_scalar(record["Ntx"], "Ntx", 1, 3)
        value = np.asarray(record["csi"])
        if nrx != 3 or ntx != 3 or value.shape != (3, 3, 30) or not np.iscomplexobj(value):
            raise ValueError("Expected complex stored [3,3,30] and Ntx=Nrx=3")
        if not np.isfinite(value).all():
            raise ValueError("Non-finite CSI")
        perm = tuple(integer_scalar(v, "perm", 1, 3) for v in np.asarray(record["perm"]).ravel())
        if sorted(perm) != [1, 2, 3]:
            raise ValueError("Malformed perm field")
        # Store selection is provisional until export provenance is resolved.
        # Never apply perm solely because it is not the identity.
        csi.append(value[0, :, :].reshape(90))
        timestamps.append(integer_scalar(record["timestamp_low"], "timestamp_low", -(2**31), 2**32 - 1))
        perms.append(perm)
    csi = np.asarray(csi, dtype=np.complex128)
    elapsed, stats = relative_microseconds(timestamps)
    stats.update({"packets": len(csi), "raw_csi_shape": [len(csi), 3, 3, 30],
                  "selected_csi_shape": list(csi.shape), "selected_zero_amplitude_values": int(np.sum(np.abs(csi) == 0)),
                  "perm_counts": {",".join(map(str, k)): v for k, v in Counter(perms).items()},
                  "perm_reapplied": False, "physical_axis_export_confirmed": False})
    return csi, elapsed, stats


def reconstruct(csi, elapsed):
    """Historical non-circular phase arithmetic, native-time shared grid."""
    csi = np.asarray(csi, dtype=np.complex128)
    elapsed = np.asarray(elapsed, dtype=np.int64)
    if csi.shape != (len(elapsed), 90) or not np.isfinite(csi).all():
        raise ValueError("Expected finite selected CSI [packets,90]")
    if elapsed[0] != 0 or elapsed[-1] <= 0 or np.any(np.diff(elapsed) < 0):
        raise ValueError("Invalid elapsed times")
    times, inverse, counts = np.unique(elapsed, return_inverse=True, return_counts=True)
    grid = np.linspace(0.0, float(elapsed[-1]), 850)
    outputs = []
    for values in (np.abs(csi), np.angle(csi)):
        means = np.zeros((len(times), 90), dtype=np.float64)
        np.add.at(means, inverse, values)
        means /= counts[:, None]
        sampled = np.column_stack([np.interp(grid, times, means[:, col]) for col in range(90)])
        outputs.append(sampled)
    amp = outputs[0].astype(np.float32)
    phase = np.unwrap(outputs[1], axis=-1).astype(np.float32)
    if not np.isfinite(amp).all() or not np.isfinite(phase).all():
        raise ValueError("Non-finite or float32 overflow in output")
    return amp, phase
