from __future__ import annotations

import awkward as ak
import numpy as np

from autonomous_allhad import flat_ntuple_worker as flat


def test_compact_2024_storage_keeps_only_gen_weight_as_float64() -> None:
    row = {
        **{name: 1 for name in flat.INT64_FIELDS},
        **{name: 1 for name in flat.INT32_FIELDS},
        **{name: 1.25 for name in flat.FLOAT_FIELDS},
        **{name: True for name in flat.BOOL_FIELDS},
        **{name: [1.25, 2.5] for name in flat.VECTOR_FLOAT_FIELDS},
        **{name: [1, 2] for name in flat.VECTOR_INT_FIELDS},
    }
    try:
        flat.configure_float_storage(float32=True, keep_float64={"gen_weight"})
        types = flat.branch_types()
        arrays = flat.rows_to_arrays([row])

        assert types["gen_weight"] is np.float64
        assert types["met"] is np.float32
        assert types["good_jet_pt"] == "var * float32"
        assert arrays["gen_weight"].dtype == np.dtype("float64")
        assert arrays["met"].dtype == np.dtype("float32")
        assert ak.to_numpy(ak.flatten(arrays["good_jet_pt"])).dtype == np.dtype(
            "float32"
        )
    finally:
        flat.configure_float_storage(float32=False)


def test_float_storage_policy_rejects_unknown_branches() -> None:
    try:
        with np.testing.assert_raises_regex(RuntimeError, "unknown branches"):
            flat.configure_float_storage(
                float32=True,
                keep_float64={"not_a_branch"},
            )
    finally:
        flat.configure_float_storage(float32=False)
