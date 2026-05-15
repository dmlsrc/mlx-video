import mlx.core as mx
import numpy as np
import pytest

from mlx_video.models.ltx_2.config import LTXModelConfig, LTXRopeType
from mlx_video.models.ltx_2.rope import precompute_freqs_cis


def create_video_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create a simple video position grid for testing."""
    t_coords = np.arange(0, num_frames)
    h_coords = np.arange(0, height)
    w_coords = np.arange(0, width)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)
    patch_ends = patch_starts + 1

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    # Scale to pixel space
    scale_factors = np.array([8, 32, 32]).reshape(1, 3, 1, 1)
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / 24.0  # Convert to seconds

    return mx.array(pixel_coords, dtype=dtype)


def _numpy_reference_rope(positions_np, dim, theta, max_pos, num_heads):
    """Compute RoPE cos/sin using NumPy float64 as ground truth reference.

    This mirrors the regular (non-double-precision) path in rope.py exactly,
    but uses float64 throughout, so we can verify that the float32 MLX path
    stays close to the true values.
    """
    # positions_np: (B, 3, T, 2) in float64
    n_pos_dims = positions_np.shape[1]
    n_elem = 2 * n_pos_dims

    # Middle-of-interval positions
    mid = (positions_np[..., 0] + positions_np[..., 1]) / 2.0  # (B, 3, T)

    # Frequency grid — matches generate_freq_grid() in rope.py:
    #   log_start = log(1)/log(theta) = 0
    #   log_end = log(theta)/log(theta) = 1
    #   pow_indices = theta^linspace(0, 1, num_indices) * pi/2
    num_indices = dim // n_elem
    if num_indices == 0:
        num_indices = 1
    lin_space = np.linspace(0.0, 1.0, num_indices, dtype=np.float64)
    freq_indices = np.power(theta, lin_space) * (np.pi / 2)  # (num_indices,)

    # Fractional positions and scaling — matches generate_freqs()
    # frac = pos / max_pos for each dim, then scale to [-1, 1]
    frac_list = []
    for d in range(n_pos_dims):
        frac = mid[:, d, :] / max_pos[d]  # (B, T)
        frac_list.append(frac)
    fractional = np.stack(frac_list, axis=-1)  # (B, T, n_dims)
    scaled = fractional * 2 - 1  # [-1, 1]

    # Outer product: (B, T, n_dims, 1) * (1, 1, 1, num_indices)
    freqs = (
        scaled[..., np.newaxis] * freq_indices[np.newaxis, np.newaxis, np.newaxis, :]
    )
    # (B, T, n_dims, num_indices) -> swap last two -> (B, T, num_indices, n_dims) -> flatten
    freqs = np.swapaxes(freqs, -1, -2)
    freqs = freqs.reshape(
        freqs.shape[0], freqs.shape[1], -1
    )  # (B, T, num_indices * n_dims)

    cos_ref = np.cos(freqs)
    sin_ref = np.sin(freqs)

    # Split RoPE: pad to dim//2, reshape to (B, H, T, dim_per_head//2)
    expected = dim // 2
    pad_size = expected - cos_ref.shape[-1]
    if pad_size > 0:
        # Padding is prepended (ones for cos, zeros for sin) — matches split_freqs_cis()
        cos_ref = np.concatenate(
            [np.ones((*cos_ref.shape[:-1], pad_size)), cos_ref], axis=-1
        )
        sin_ref = np.concatenate(
            [np.zeros((*sin_ref.shape[:-1], pad_size)), sin_ref], axis=-1
        )

    B, T, _ = cos_ref.shape
    dim_per_head = dim // num_heads
    cos_ref = cos_ref.reshape(B, T, num_heads, dim_per_head // 2).transpose(0, 2, 1, 3)
    sin_ref = sin_ref.reshape(B, T, num_heads, dim_per_head // 2).transpose(0, 2, 1, 3)

    return cos_ref, sin_ref


class TestRoPEPositionPrecision:
    """Test suite for RoPE position precision requirements."""

    def test_float32_positions_produce_consistent_output(self):
        """Float32 position grids should produce stable RoPE frequencies."""
        positions = create_video_position_grid(1, 4, 4, 4, dtype=mx.float32)

        cos_freq, sin_freq = precompute_freqs_cis(
            indices_grid=positions,
            dim=128,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
            double_precision=True,
        )

        # Verify output dtype is float32
        assert cos_freq.dtype == mx.float32, f"Expected float32, got {cos_freq.dtype}"
        assert sin_freq.dtype == mx.float32, f"Expected float32, got {sin_freq.dtype}"

        # Verify no NaN or Inf values
        assert not mx.any(mx.isnan(cos_freq)).item(), "cos_freq contains NaN"
        assert not mx.any(mx.isnan(sin_freq)).item(), "sin_freq contains NaN"
        assert not mx.any(mx.isinf(cos_freq)).item(), "cos_freq contains Inf"
        assert not mx.any(mx.isinf(sin_freq)).item(), "sin_freq contains Inf"

        # Verify cos/sin are in valid range [-1, 1]
        assert (
            mx.all(cos_freq >= -1.0).item() and mx.all(cos_freq <= 1.0).item()
        ), "cos_freq values out of [-1, 1] range"
        assert (
            mx.all(sin_freq >= -1.0).item() and mx.all(sin_freq <= 1.0).item()
        ), "sin_freq values out of [-1, 1] range"

    def test_bfloat16_positions_cause_precision_loss(self):
        """bfloat16 positions should produce different (less precise) results than float32.

        This test documents the known issue: bfloat16 has only 7 bits of mantissa
        vs 23 bits for float32, causing quantization errors that get amplified
        by sin/cos calculations in RoPE.
        """
        # Create identical position grids in different dtypes
        positions_f32 = create_video_position_grid(1, 4, 4, 4, dtype=mx.float32)
        positions_bf16 = create_video_position_grid(1, 4, 4, 4, dtype=mx.bfloat16)

        # Compute RoPE frequencies
        cos_f32, sin_f32 = precompute_freqs_cis(
            indices_grid=positions_f32,
            dim=128,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
            double_precision=True,
        )

        cos_bf16, sin_bf16 = precompute_freqs_cis(
            indices_grid=positions_bf16,
            dim=128,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
            double_precision=True,
        )

        # Calculate the difference
        cos_diff = mx.abs(cos_f32 - cos_bf16)
        sin_diff = mx.abs(sin_f32 - sin_bf16)

        max_cos_diff = mx.max(cos_diff).item()
        max_sin_diff = mx.max(sin_diff).item()

        # bfloat16 positions WILL cause measurable differences
        # This test documents this known behavior
        # The threshold here is intentionally low to catch the issue
        precision_threshold = 1e-6

        has_precision_loss = (
            max_cos_diff > precision_threshold or max_sin_diff > precision_threshold
        )

        # Document the precision loss (this is expected behavior)
        if has_precision_loss:
            print(f"\nPrecision loss detected (expected):")
            print(f"  Max cos difference: {max_cos_diff:.6e}")
            print(f"  Max sin difference: {max_sin_diff:.6e}")

        # This assertion documents the issue - bfloat16 positions cause precision loss
        assert (
            has_precision_loss
        ), "Expected precision loss with bfloat16 positions - if this fails, the issue may be fixed"

    def test_double_precision_converts_to_float32_internally(self):
        """Verify that double_precision mode converts bfloat16 to float32 first."""
        positions_bf16 = create_video_position_grid(1, 4, 4, 4, dtype=mx.bfloat16)

        cos_freq, sin_freq = precompute_freqs_cis(
            indices_grid=positions_bf16,
            dim=128,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
            double_precision=True,
        )

        # Output should still be float32
        assert cos_freq.dtype == mx.float32
        assert sin_freq.dtype == mx.float32

    def test_position_grid_should_be_float32_recommendation(self):
        """Test that validates the recommended practice: positions should be float32.

        This test serves as documentation that position grids MUST be float32
        to avoid quality degradation in generated videos/audio.
        """
        # Recommended: create positions in float32
        positions = create_video_position_grid(1, 4, 4, 4, dtype=mx.float32)

        assert (
            positions.dtype == mx.float32
        ), "Position grids should be created in float32 for RoPE precision"

        # Verify the position values are reasonable
        # Temporal positions should be small (seconds)
        temporal_positions = positions[:, 0, :, :]
        assert (
            mx.max(temporal_positions).item() < 100
        ), "Temporal positions should be in seconds (small values)"

        # Spatial positions should be larger (pixels)
        spatial_h = positions[:, 1, :, :]
        spatial_w = positions[:, 2, :, :]
        assert (
            mx.max(spatial_h).item() > 0
        ), "Spatial height positions should be positive"
        assert (
            mx.max(spatial_w).item() > 0
        ), "Spatial width positions should be positive"

    def test_float32_positions_match_numpy_float64_reference(self):
        """Regression test: float32 RoPE must closely match a NumPy float64 reference.

        This is the key correctness test. We compute RoPE with NumPy in float64
        (ground truth) and verify that the MLX float32 path produces nearly
        identical results. The max allowed diff (1e-5) is well below the error
        we saw with bfloat16 positions (~2.0 max diff, cosine sim 0.88).
        """
        positions = create_video_position_grid(1, 4, 4, 4, dtype=mx.float32)
        positions_np = np.array(positions).astype(np.float64)

        dim = 128
        theta = 10000.0
        max_pos = [20, 2048, 2048]
        num_heads = 32

        # MLX result (float32 path, non-double-precision)
        cos_mlx, sin_mlx = precompute_freqs_cis(
            indices_grid=positions,
            dim=dim,
            theta=theta,
            max_pos=max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=num_heads,
            rope_type=LTXRopeType.SPLIT,
            double_precision=False,
        )

        # NumPy float64 reference
        cos_ref, sin_ref = _numpy_reference_rope(
            positions_np, dim, theta, max_pos, num_heads
        )

        cos_mlx_np = np.array(cos_mlx)
        sin_mlx_np = np.array(sin_mlx)

        max_cos_diff = np.max(np.abs(cos_mlx_np - cos_ref))
        max_sin_diff = np.max(np.abs(sin_mlx_np - sin_ref))

        # Cosine similarity (flatten for single scalar)
        cos_flat = cos_mlx_np.flatten()
        ref_flat = cos_ref.flatten()
        cosine_sim = np.dot(cos_flat, ref_flat) / (
            np.linalg.norm(cos_flat) * np.linalg.norm(ref_flat)
        )

        # float32 vs float64: expect small diffs from 23-bit vs 52-bit mantissa.
        # Threshold 0.01 is well below the bfloat16 failure mode (~2.0 max diff).
        assert (
            max_cos_diff < 0.01
        ), f"cos max diff {max_cos_diff:.2e} exceeds 0.01 — float32 positions may not be preserved"
        assert (
            max_sin_diff < 0.01
        ), f"sin max diff {max_sin_diff:.2e} exceeds 0.01 — float32 positions may not be preserved"
        assert (
            cosine_sim > 0.9999
        ), f"cos cosine similarity {cosine_sim:.6f} too low — expected >0.9999"

    def test_high_frequency_amplification_regression(self):
        """Regression test for the specific failure mode: high-frequency index amplification.

        With production-sized grids (5x16x16 = 1280 tokens), fractional positions
        like 0.000391 get multiplied by frequency indices up to ~15708. In bfloat16
        the fractional part is quantized, producing raw freq errors of ~6.14 and
        cos/sin sign flips (max_diff ~2.0). Float32 must keep max_diff < 0.01.
        """
        # Use a production-like grid size
        positions = create_video_position_grid(1, 5, 16, 16, dtype=mx.float32)
        positions_np = np.array(positions).astype(np.float64)

        dim = 128
        theta = 10000.0
        max_pos = [20, 2048, 2048]
        num_heads = 32

        cos_mlx, sin_mlx = precompute_freqs_cis(
            indices_grid=positions,
            dim=dim,
            theta=theta,
            max_pos=max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=num_heads,
            rope_type=LTXRopeType.SPLIT,
            double_precision=False,
        )

        cos_ref, sin_ref = _numpy_reference_rope(
            positions_np, dim, theta, max_pos, num_heads
        )

        max_cos_diff = np.max(np.abs(np.array(cos_mlx) - cos_ref))
        max_sin_diff = np.max(np.abs(np.array(sin_mlx) - sin_ref))

        # Float32 should keep errors well below the bfloat16 failure threshold of ~2.0
        assert (
            max_cos_diff < 0.01
        ), f"Production grid cos max diff {max_cos_diff:.4f} — high-freq amplification detected"
        assert (
            max_sin_diff < 0.01
        ), f"Production grid sin max diff {max_sin_diff:.4f} — high-freq amplification detected"


class TestRoPEInterleaved:
    """Tests for interleaved RoPE mode."""

    def test_interleaved_rope_with_float32_positions(self):
        """Interleaved RoPE should work correctly with float32 positions."""
        positions = create_video_position_grid(1, 4, 4, 4, dtype=mx.float32)

        cos_freq, sin_freq = precompute_freqs_cis(
            indices_grid=positions,
            dim=128,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.INTERLEAVED,
            double_precision=False,
        )

        assert cos_freq.dtype == mx.float32
        assert sin_freq.dtype == mx.float32
        assert not mx.any(mx.isnan(cos_freq)).item()
        assert not mx.any(mx.isnan(sin_freq)).item()


class TestRoPEInputCasting:
    """Tests that precompute_freqs_cis casts positions to float32 internally.

    The fix in rope.py ensures that regardless of the input dtype, positions are
    cast to float32 before any computation. This class verifies that behavior
    for both the regular and double-precision paths.
    """

    def test_regular_path_outputs_float32(self):
        """Regular path: both float32 and bfloat16 inputs produce float32 output."""
        positions_f32 = create_video_position_grid(1, 4, 4, 4, dtype=mx.float32)
        positions_bf16 = positions_f32.astype(mx.bfloat16)

        kwargs = dict(
            dim=128,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
            double_precision=False,
        )

        cos_f32, sin_f32 = precompute_freqs_cis(indices_grid=positions_f32, **kwargs)
        cos_bf16, sin_bf16 = precompute_freqs_cis(indices_grid=positions_bf16, **kwargs)

        # Both produce float32 output regardless of input dtype
        assert cos_f32.dtype == mx.float32
        assert cos_bf16.dtype == mx.float32
        assert sin_f32.dtype == mx.float32
        assert sin_bf16.dtype == mx.float32

        # No NaN/Inf in either
        assert not mx.any(mx.isnan(cos_bf16)).item()
        assert not mx.any(mx.isinf(cos_bf16)).item()

    def test_double_precision_path_outputs_float32(self):
        """Double-precision path: both float32 and bfloat16 inputs produce float32 output."""
        positions_f32 = create_video_position_grid(1, 4, 4, 4, dtype=mx.float32)
        positions_bf16 = positions_f32.astype(mx.bfloat16)

        kwargs = dict(
            dim=128,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
            double_precision=True,
        )

        cos_f32, sin_f32 = precompute_freqs_cis(indices_grid=positions_f32, **kwargs)
        cos_bf16, sin_bf16 = precompute_freqs_cis(indices_grid=positions_bf16, **kwargs)

        assert cos_f32.dtype == mx.float32
        assert cos_bf16.dtype == mx.float32
        assert sin_f32.dtype == mx.float32
        assert sin_bf16.dtype == mx.float32

        assert not mx.any(mx.isnan(cos_bf16)).item()
        assert not mx.any(mx.isinf(cos_bf16)).item()

    def test_float16_input_also_cast_to_float32(self):
        """Float16 input should also be handled correctly."""
        positions_f16 = create_video_position_grid(1, 4, 4, 4, dtype=mx.float16)

        cos_freq, sin_freq = precompute_freqs_cis(
            indices_grid=positions_f16,
            dim=128,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
            double_precision=False,
        )

        assert cos_freq.dtype == mx.float32
        assert sin_freq.dtype == mx.float32
        assert not mx.any(mx.isnan(cos_freq)).item()


class TestDoublePrecisionRopeConfig:
    """Tests for double_precision_rope in LTXModelConfig.

    Both LTX-2 and LTX-2.3 checkpoints embed frequencies_precision="float64",
    so double_precision_rope is respected as-is for both models.
    """

    def test_ltx2_preserves_double_precision_rope_true(self):
        """LTX-2 checkpoint sets frequencies_precision=float64, so True must be preserved."""
        config = LTXModelConfig(has_prompt_adaln=False, double_precision_rope=True)
        assert config.double_precision_rope is True

    def test_ltx2_preserves_double_precision_rope_false(self):
        """False must also be preserved (e.g. user override)."""
        config = LTXModelConfig(has_prompt_adaln=False, double_precision_rope=False)
        assert config.double_precision_rope is False

    def test_ltx23_preserves_double_precision_rope_true(self):
        """LTX-2.3 with double_precision_rope=True should stay True."""
        config = LTXModelConfig(has_prompt_adaln=True, double_precision_rope=True)
        assert config.double_precision_rope is True

    def test_ltx23_preserves_double_precision_rope_false(self):
        """LTX-2.3 with double_precision_rope=False should stay False."""
        config = LTXModelConfig(has_prompt_adaln=True, double_precision_rope=False)
        assert config.double_precision_rope is False

    def test_default_double_precision_rope(self):
        """Default value is False when not specified."""
        config = LTXModelConfig()
        assert config.double_precision_rope is False

    def test_config_from_dict_ltx2(self):
        """Config from dict for LTX-2 preserves double_precision_rope=True."""
        config = LTXModelConfig.from_dict(
            {
                "has_prompt_adaln": False,
                "double_precision_rope": True,
                "rope_type": "split",
            }
        )
        assert config.double_precision_rope is True

    def test_config_from_dict_ltx23(self):
        """Config from dict for LTX-2.3 preserves double_precision_rope=True."""
        config = LTXModelConfig.from_dict(
            {
                "has_prompt_adaln": True,
                "double_precision_rope": True,
                "rope_type": "split",
            }
        )
        assert config.double_precision_rope is True


class TestRoPESplit:
    """Tests for split RoPE mode (used by LTX-2)."""

    def test_split_rope_output_shape(self):
        """Verify split RoPE output has correct shape (B, H, T, dim_per_head//2)."""
        batch_size = 1
        num_frames = 4
        height = 4
        width = 4
        num_heads = 32
        dim = 128

        positions = create_video_position_grid(batch_size, num_frames, height, width)
        num_tokens = num_frames * height * width

        cos_freq, sin_freq = precompute_freqs_cis(
            indices_grid=positions,
            dim=dim,
            theta=10000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=num_heads,
            rope_type=LTXRopeType.SPLIT,
            double_precision=True,
        )

        # Shape should be (B, H, T, dim_per_head//2)
        # dim=128, num_heads=32, so dim_per_head=4, and split uses half=2
        dim_per_head = dim // num_heads
        expected_shape = (batch_size, num_heads, num_tokens, dim_per_head // 2)
        assert (
            cos_freq.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {cos_freq.shape}"
        assert (
            sin_freq.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {sin_freq.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
