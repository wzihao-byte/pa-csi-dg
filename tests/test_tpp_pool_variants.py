from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TPP_ROOT = REPO_ROOT / "experiments" / "tpp_isolated"
if str(TPP_ROOT) not in sys.path:
    sys.path.insert(0, str(TPP_ROOT))

from tpp_models import IsolatedTemporalPyramidPool


@pytest.mark.parametrize("pool_type", ["max", "avg", "mixed"])
def test_tpp_pool_variants_keep_output_shape(pool_type: str) -> None:
    pool = IsolatedTemporalPyramidPool(
        input_dim=6,
        output_channels=4,
        kernel_sizes=[3, 5],
        levels=[1, 2, 4],
        dropout=0.0,
        pool_type=pool_type,
    )

    output = pool(torch.randn(3, 24, 6))

    assert output.shape == (3, 8)


def test_default_max_pool_matches_original_projection_path() -> None:
    torch.manual_seed(7)
    pool = IsolatedTemporalPyramidPool(
        input_dim=5,
        output_channels=3,
        kernel_sizes=[3, 5],
        levels=[1, 2, 4],
        dropout=0.0,
    )
    inputs = torch.randn(2, 20, 5)

    actual = pool(inputs)
    responses = [pool.activation(conv(inputs.transpose(1, 2))) for conv in pool.convs]
    pyramid = torch.cat(
        [
            F.adaptive_max_pool1d(response, output_size=level).flatten(start_dim=1)
            for response in responses
            for level in pool.levels
        ],
        dim=1,
    )
    expected = pool.projection(pyramid)

    torch.testing.assert_close(actual, expected)


def test_zero_scale_residual_returns_original_global_max_feature() -> None:
    torch.manual_seed(11)
    pool = IsolatedTemporalPyramidPool(
        input_dim=5,
        output_channels=3,
        kernel_sizes=[3, 5],
        levels=[1, 2, 4],
        dropout=0.0,
        preserve_global=True,
        pyramid_scale=0.0,
    )
    inputs = torch.randn(2, 20, 5)

    actual = pool(inputs)
    expected = torch.cat(
        [
            F.adaptive_max_pool1d(
                pool.activation(conv(inputs.transpose(1, 2))), output_size=1
            ).flatten(start_dim=1)
            for conv in pool.convs
        ],
        dim=1,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pool_type": "median"}, "Unsupported TPP pool_type"),
        ({"mixed_max_weight": 1.5}, "mixed_max_weight"),
        ({"pyramid_scale": -0.1}, "pyramid_scale"),
    ],
)
def test_invalid_c3_recovery_pool_settings_fail_fast(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IsolatedTemporalPyramidPool(
            input_dim=5,
            output_channels=3,
            kernel_sizes=[3],
            levels=[1, 2],
            **kwargs,
        )
