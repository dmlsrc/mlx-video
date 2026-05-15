"""Convert LTX-2/2.3 safetensors to MLX directory layout.

Converts from the single-file format (e.g. Lightricks/LTX-2/ltx-2-19b-distilled.safetensors
or Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors) to the modular directory structure:

    output/
    ├── transformer/          # DiT transformer weights (sharded)
    │   ├── config.json
    │   ├── model-00001-of-N.safetensors
    │   └── model.safetensors.index.json
    ├── vae/
    │   ├── decoder/          # Video VAE decoder
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   └── encoder/          # Video VAE encoder
    │       ├── config.json
    │       └── model.safetensors
    ├── audio_vae/
    │   ├── decoder/          # Audio VAE decoder
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   └── encoder/          # Audio VAE encoder
    │       ├── config.json
    │       └── model.safetensors
    ├── vocoder/              # Audio vocoder
    │   ├── config.json
    │   └── model.safetensors
    └── text_projections/     # Text projection connectors
        └── model.safetensors

Usage:
    # From HF repo ID (downloads if not cached)
    python -m mlx_video.models.ltx_2.convert --source Lightricks/LTX-2 --output LTX-2-distilled --variant distilled
    python -m mlx_video.models.ltx_2.convert --source Lightricks/LTX-2.3 --output LTX-2.3-distilled --variant distilled

    # From HF repo ID using only local cache (no network)
    python -m mlx_video.models.ltx_2.convert --source Lightricks/LTX-2.3 --output LTX-2.3-distilled --variant distilled --cached-only

    # From local folder containing the monolithic safetensors
    python -m mlx_video.models.ltx_2.convert --source ./Lightricks-LTX-2/ --output LTX-2-distilled --variant distilled

    # From a direct safetensors file path
    python -m mlx_video.models.ltx_2.convert --source ./ltx-2-19b-distilled.safetensors --output LTX-2-distilled --variant distilled
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import mlx.core as mx

from mlx_video.models.ltx_2.checkpoint import (
    MONOLITHIC_PATTERN,
    UPSCALER_PATTERN,
    find_cached_hf_snapshot,
    read_embedded_config,
    resolve_source,
)

GEMMA_REPO_ID = "google/gemma-3-12b-it"

# ─── Key prefix routing ──────────────────────────────────────────────────────

TRANSFORMER_PREFIX = "model.diffusion_model."
VAE_DECODER_PREFIX = "vae.decoder."
VAE_ENCODER_PREFIX = "vae.encoder."
VAE_STATS_PREFIX = "vae.per_channel_statistics."
AUDIO_DECODER_PREFIX = "audio_vae.decoder."
AUDIO_ENCODER_PREFIX = "audio_vae.encoder."
AUDIO_STATS_PREFIX = "audio_vae.per_channel_statistics."
VOCODER_PREFIX = "vocoder."
TEXT_PROJ_PREFIX = "text_embedding_projection."
VIDEO_CONNECTOR_PREFIX = "model.diffusion_model.video_embeddings_connector."
AUDIO_CONNECTOR_PREFIX = "model.diffusion_model.audio_embeddings_connector."


# ─── Sanitization functions ──────────────────────────────────────────────────


def sanitize_transformer(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize transformer keys: strip prefix, rename layers, cast to bfloat16."""
    sanitized = {}
    for key, value in weights.items():
        if not key.startswith(TRANSFORMER_PREFIX):
            continue
        # Skip connector weights (they go to text_projections)
        if "audio_embeddings_connector" in key or "video_embeddings_connector" in key:
            continue

        new_key = key[len(TRANSFORMER_PREFIX):]
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
        new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
        new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
        new_key = new_key.replace(".linear_1.", ".linear1.")
        new_key = new_key.replace(".linear_2.", ".linear2.")

        # Cast all weights to bfloat16 (matches MLX model loading behavior)
        if value.dtype != mx.bfloat16:
            value = value.astype(mx.bfloat16)

        sanitized[new_key] = value
    return sanitized


def sanitize_vae_decoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE decoder keys: strip prefix, transpose Conv3d, wrap .conv."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if key.startswith(VAE_STATS_PREFIX):
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                continue
        elif key.startswith(VAE_DECODER_PREFIX):
            new_key = key[len(VAE_DECODER_PREFIX):]
        else:
            continue

        # Conv3d weight transpose: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
        if ".conv.weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Wrap .conv.weight -> .conv.conv.weight (CausalConv3d wrapper)
        if ".conv.weight" in new_key or ".conv.bias" in new_key:
            if ".conv.conv.weight" not in new_key and ".conv.conv.bias" not in new_key:
                new_key = new_key.replace(".conv.weight", ".conv.conv.weight")
                new_key = new_key.replace(".conv.bias", ".conv.conv.bias")

        sanitized[new_key] = value
    return sanitized


def sanitize_vae_encoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE encoder keys: strip prefix, transpose Conv3d/Conv2d."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if "position_ids" in key:
            continue

        if key.startswith(VAE_STATS_PREFIX):
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                continue
            # Per-channel statistics must stay float32 for precision
            if value.dtype != mx.float32:
                value = value.astype(mx.float32)
        elif key.startswith(VAE_ENCODER_PREFIX):
            new_key = key[len(VAE_ENCODER_PREFIX):]
        else:
            continue

        # Conv3d: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_audio_decoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize audio VAE decoder keys: strip prefix, transpose Conv2d."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if key.startswith(AUDIO_DECODER_PREFIX):
            new_key = key[len(AUDIO_DECODER_PREFIX):]
        elif key.startswith(AUDIO_STATS_PREFIX):
            if "mean-of-means" in key:
                new_key = "per_channel_statistics.mean_of_means"
            elif "std-of-means" in key:
                new_key = "per_channel_statistics.std_of_means"
            else:
                continue
        else:
            continue

        # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_audio_encoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize audio VAE encoder keys: strip prefix, transpose Conv2d."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if key.startswith(AUDIO_ENCODER_PREFIX):
            new_key = key[len(AUDIO_ENCODER_PREFIX):]
        elif key.startswith(AUDIO_STATS_PREFIX):
            if "mean-of-means" in key:
                new_key = "per_channel_statistics.mean_of_means"
            elif "std-of-means" in key:
                new_key = "per_channel_statistics.std_of_means"
            else:
                continue
        elif key == "latents_mean":
            new_key = "per_channel_statistics.mean_of_means"
        elif key == "latents_std":
            new_key = "per_channel_statistics.std_of_means"
        else:
            continue

        # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_vocoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize vocoder keys: strip prefix, transpose Conv1d/ConvTranspose1d."""
    sanitized = {}
    for key, value in weights.items():
        if not key.startswith(VOCODER_PREFIX):
            continue

        new_key = key[len(VOCODER_PREFIX):]

        # Handle Conv1d/ConvTranspose1d weight shape conversion
        if "weight" in new_key and value.ndim == 3:
            if "ups" in new_key:
                # ConvTranspose1d: PyTorch (in_ch, out_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                value = mx.transpose(value, (1, 2, 0))
            else:
                # Conv1d: PyTorch (out_ch, in_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                value = mx.transpose(value, (0, 2, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_connector_key(key: str) -> str:
    """Sanitize connector sub-key names."""
    key = key.replace(".ff.net.0.proj.", ".ff.proj_in.")
    key = key.replace(".ff.net.2.", ".ff.proj_out.")
    key = key.replace(".to_out.0.", ".to_out.")
    return key


def extract_text_projections(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Extract text projection weights (aggregate_embed + connectors).

    Handles both LTX-2 (aggregate_embed.weight) and LTX-2.3
    (video_aggregate_embed.*, audio_aggregate_embed.*) formats.
    """
    extracted = {}

    # aggregate_embed weights (text_embedding_projection.*)
    for key, value in weights.items():
        if key.startswith(TEXT_PROJ_PREFIX):
            new_key = key[len(TEXT_PROJ_PREFIX):]
            extracted[new_key] = value

    # video_embeddings_connector
    for key, value in weights.items():
        if key.startswith(VIDEO_CONNECTOR_PREFIX):
            suffix = key[len(VIDEO_CONNECTOR_PREFIX):]
            new_key = "video_embeddings_connector." + sanitize_connector_key(suffix)
            extracted[new_key] = value

    # audio_embeddings_connector
    for key, value in weights.items():
        if key.startswith(AUDIO_CONNECTOR_PREFIX):
            suffix = key[len(AUDIO_CONNECTOR_PREFIX):]
            new_key = "audio_embeddings_connector." + sanitize_connector_key(suffix)
            extracted[new_key] = value

    return extracted


# ─── Saving utilities ─────────────────────────────────────────────────────────


def save_sharded(
    weights: Dict[str, mx.array],
    output_dir: Path,
    max_shard_size_bytes: int = 5 * 1024 * 1024 * 1024,  # 5GB per shard
):
    """Save weights as sharded safetensors with an index file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort keys for deterministic output
    sorted_keys = sorted(weights.keys())

    # Calculate total size
    total_size = sum(weights[k].nbytes for k in sorted_keys)

    # Determine sharding
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted_keys:
        tensor = weights[key]
        tensor_size = tensor.nbytes

        if current_size + tensor_size > max_shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}

    for i, shard in enumerate(shards):
        if num_shards == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"

        mx.save_safetensors(str(output_dir / filename), shard)

        for key in shard:
            weight_map[key] = filename

    # Write index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    return num_shards


def save_single(weights: Dict[str, mx.array], output_dir: Path):
    """Save weights as a single safetensors file with an index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)

    # Also write index for consistency
    total_size = sum(v.nbytes for v in weights.values())
    weight_map = {k: "model.safetensors" for k in sorted(weights.keys())}
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def save_config(config: dict, output_dir: Path):
    """Save config.json to a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


# ─── Config inference ─────────────────────────────────────────────────────────


def infer_transformer_config(weights: Dict[str, mx.array], embedded: dict) -> dict:
    """Infer transformer config from embedded checkpoint config and weight keys."""
    tcfg = embedded.get("transformer", {})

    # has_prompt_adaln is not in the embedded config — detect from weight keys
    has_prompt_adaln = any("prompt_adaln_single" in k for k in weights)

    config = {
        "model_type": "ltx av model",
        "num_attention_heads": tcfg.get("num_attention_heads", 32),
        "attention_head_dim": tcfg.get("attention_head_dim", 128),
        "in_channels": tcfg.get("in_channels", 128),
        "out_channels": tcfg.get("out_channels", 128),
        "num_layers": tcfg.get("num_layers", 48),
        "cross_attention_dim": tcfg.get("cross_attention_dim", 4096),
        "caption_channels": tcfg.get("caption_channels", 3840),
        "audio_num_attention_heads": tcfg.get("audio_num_attention_heads", 32),
        "audio_attention_head_dim": tcfg.get("audio_attention_head_dim", 64),
        "audio_in_channels": 128,  # not in embedded; constant across versions
        "audio_out_channels": tcfg.get("audio_out_channels", 128),
        "audio_cross_attention_dim": tcfg.get("audio_cross_attention_dim", 2048),
        "audio_caption_channels": tcfg.get("caption_channels", 3840),
        "positional_embedding_theta": tcfg.get("positional_embedding_theta", 10000.0),
        "positional_embedding_max_pos": tcfg.get("positional_embedding_max_pos", [20, 2048, 2048]),
        "audio_positional_embedding_max_pos": tcfg.get("audio_positional_embedding_max_pos", [20]),
        "use_middle_indices_grid": tcfg.get("use_middle_indices_grid", True),
        "rope_type": tcfg.get("rope_type", "split"),
        "double_precision_rope": tcfg.get("frequencies_precision") == "float64",
        "timestep_scale_multiplier": tcfg.get("timestep_scale_multiplier", 1000),
        "av_ca_timestep_scale_multiplier": int(tcfg.get("av_ca_timestep_scale_multiplier", 1000)),
        "norm_eps": tcfg.get("norm_eps", 1e-6),
        "attention_type": tcfg.get("attention_type", "default"),
    }

    if has_prompt_adaln:
        config["has_prompt_adaln"] = True

    return config


def infer_vae_decoder_config(embedded: dict) -> dict:
    """Infer VAE decoder config from embedded checkpoint config."""
    vcfg = embedded.get("vae", {})
    return {
        "ch": 128,
        "ch_mult": [1, 2, 4],
        "dropout": 0.0,
        "num_res_blocks": 2,
        "out_ch": 2,
        "resolution": 256,
        "spatial_padding_mode": vcfg.get("spatial_padding_mode", "zeros"),
        "timestep_conditioning": vcfg.get("timestep_conditioning", False),
        "z_channels": 8,
    }


def infer_vae_encoder_config(embedded: dict) -> dict:
    """Infer VAE encoder config from embedded checkpoint config."""
    vcfg = embedded.get("vae", {})
    encoder_blocks = vcfg.get("encoder_blocks", [
        ["res_x", {"num_layers": 4}],
        ["compress_space_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 6}],
        ["compress_time_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 6}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 2}],
        ["compress_all_res", {"multiplier": 2}],
        ["res_x", {"num_layers": 2}],
    ])
    return {
        "convolution_dimensions": vcfg.get("dims", 3),
        "encoder_blocks": encoder_blocks,
        "encoder_spatial_padding_mode": vcfg.get("spatial_padding_mode", "zeros"),
        "in_channels": vcfg.get("in_channels", 3),
        "latent_log_var": vcfg.get("latent_log_var", "uniform"),
        "norm_layer": vcfg.get("norm_layer", "pixel_norm"),
        "out_channels": vcfg.get("latent_channels", 128),
        "patch_size": vcfg.get("patch_size", 4),
    }


def infer_audio_vae_config(embedded: dict) -> dict:
    """Infer audio VAE decoder config from embedded checkpoint config."""
    avae = embedded.get("audio_vae", {})
    ddconfig = avae.get("model", {}).get("params", {}).get("ddconfig", {})
    stft = avae.get("preprocessing", {}).get("stft", {})
    sampling_rate = avae.get("model", {}).get("params", {}).get("sampling_rate", 16000)

    return {
        "attn_resolutions": ddconfig.get("attn_resolutions", []),
        "attn_type": "vanilla",
        "causality_axis": ddconfig.get("causality_axis", "height"),
        "ch": ddconfig.get("ch", 128),
        "ch_mult": ddconfig.get("ch_mult", [1, 2, 4]),
        "dropout": ddconfig.get("dropout", 0.0),
        "give_pre_end": False,
        "is_causal": stft.get("causal", True),
        "mel_bins": ddconfig.get("mel_bins", 64),
        "mel_hop_length": stft.get("hop_length", 160),
        "mid_block_add_attention": ddconfig.get("mid_block_add_attention", False),
        "norm_type": ddconfig.get("norm_type", "pixel"),
        "num_res_blocks": ddconfig.get("num_res_blocks", 2),
        "out_ch": ddconfig.get("out_ch", 2),
        "resamp_with_conv": True,
        "resolution": ddconfig.get("resolution", 256),
        "sample_rate": sampling_rate,
        "tanh_out": False,
        "z_channels": ddconfig.get("z_channels", 8),
    }


def infer_audio_encoder_config(embedded: dict) -> dict:
    """Infer audio VAE encoder config from embedded checkpoint config."""
    avae = embedded.get("audio_vae", {})
    ddconfig = avae.get("model", {}).get("params", {}).get("ddconfig", {})
    stft = avae.get("preprocessing", {}).get("stft", {})
    sampling_rate = avae.get("model", {}).get("params", {}).get("sampling_rate", 16000)

    return {
        "attn_resolutions": ddconfig.get("attn_resolutions", []),
        "attn_type": "vanilla",
        "causality_axis": ddconfig.get("causality_axis", "height"),
        "ch": ddconfig.get("ch", 128),
        "ch_mult": ddconfig.get("ch_mult", [1, 2, 4]),
        "double_z": ddconfig.get("double_z", True),
        "dropout": ddconfig.get("dropout", 0.0),
        "in_channels": ddconfig.get("in_channels", 2),
        "is_causal": stft.get("causal", True),
        "mel_bins": ddconfig.get("mel_bins", 64),
        "mel_hop_length": stft.get("hop_length", 160),
        "mid_block_add_attention": ddconfig.get("mid_block_add_attention", False),
        "n_fft": stft.get("filter_length", 1024),
        "norm_type": ddconfig.get("norm_type", "pixel"),
        "num_res_blocks": ddconfig.get("num_res_blocks", 2),
        "resamp_with_conv": True,
        "resolution": ddconfig.get("resolution", 256),
        "sample_rate": sampling_rate,
        "z_channels": ddconfig.get("z_channels", 8),
    }


def infer_vocoder_config(embedded: dict) -> dict:
    """Infer vocoder config from embedded checkpoint config."""
    vcfg = embedded.get("vocoder", {})

    if "vocoder" in vcfg:
        # LTX-2.3 BigVGAN: nested vocoder + bwe sub-configs
        vocoder_sub = dict(vcfg["vocoder"])
        vocoder_sub.setdefault("output_sample_rate", 16000)
        return {
            "type": "bigvgan",
            "has_bwe_generator": True,
            "vocoder": vocoder_sub,
            "bwe": dict(vcfg["bwe"]),
        }

    # LTX-2 HiFi-GAN: flat config, add output_sample_rate (not stored in embedded)
    result = dict(vcfg)
    result.setdefault("output_sample_rate", 24000)
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────


def convert(source: str, output_path: Path, variant: str = "distilled", cached_only: bool = False):
    """Convert monolithic safetensors to modular directory layout.

    Args:
        source: HF repo ID (e.g. "Lightricks/LTX-2"), local directory, or file path.
        output_path: Output directory for the modular layout.
        variant: "distilled" or "dev".
        cached_only: If True, never trigger network downloads — use local files and HF cache only.
    """
    source_path = resolve_source(source, variant, local_files_only=cached_only)
    embedded = read_embedded_config(source_path)

    print(f"Loading monolithic weights from {source_path.name}...")
    all_weights = mx.load(str(source_path))
    total_keys = len(all_weights)
    print(f"  Loaded {total_keys} keys")

    # Route keys to components
    print("\nExtracting components...")

    # 1. Transformer
    print("  [1/7] Transformer...")
    transformer_weights = sanitize_transformer(all_weights)
    num_shards = save_sharded(transformer_weights, output_path / "transformer")
    config = infer_transformer_config(transformer_weights, embedded)
    save_config(config, output_path / "transformer")
    t_params = sum(v.size for v in transformer_weights.values())
    print(f"    {len(transformer_weights)} keys, {t_params:,} params, {num_shards} shards")

    # 2. VAE Decoder
    print("  [2/7] VAE Decoder...")
    vae_decoder_weights = sanitize_vae_decoder(all_weights)
    save_single(vae_decoder_weights, output_path / "vae" / "decoder")
    config = infer_vae_decoder_config(embedded)
    save_config(config, output_path / "vae" / "decoder")
    d_params = sum(v.size for v in vae_decoder_weights.values())
    print(f"    {len(vae_decoder_weights)} keys, {d_params:,} params")

    # 3. VAE Encoder
    print("  [3/7] VAE Encoder...")
    vae_encoder_weights = sanitize_vae_encoder(all_weights)
    save_single(vae_encoder_weights, output_path / "vae" / "encoder")
    config = infer_vae_encoder_config(embedded)
    save_config(config, output_path / "vae" / "encoder")
    e_params = sum(v.size for v in vae_encoder_weights.values())
    print(f"    {len(vae_encoder_weights)} keys, {e_params:,} params")

    # 4. Audio VAE Decoder
    print("  [4/7] Audio VAE Decoder...")
    audio_decoder_weights = sanitize_audio_decoder(all_weights)
    save_single(audio_decoder_weights, output_path / "audio_vae" / "decoder")
    config = infer_audio_vae_config(embedded)
    save_config(config, output_path / "audio_vae" / "decoder")
    a_params = sum(v.size for v in audio_decoder_weights.values())
    print(f"    {len(audio_decoder_weights)} keys, {a_params:,} params")

    # 5. Audio VAE Encoder
    print("  [5/7] Audio VAE Encoder...")
    audio_encoder_weights = sanitize_audio_encoder(all_weights)
    save_single(audio_encoder_weights, output_path / "audio_vae" / "encoder")
    config = infer_audio_encoder_config(embedded)
    save_config(config, output_path / "audio_vae" / "encoder")
    ae_params = sum(v.size for v in audio_encoder_weights.values())
    print(f"    {len(audio_encoder_weights)} keys, {ae_params:,} params")

    # 6. Vocoder
    print("  [6/7] Vocoder...")
    vocoder_weights = sanitize_vocoder(all_weights)
    save_single(vocoder_weights, output_path / "vocoder")
    config = infer_vocoder_config(embedded)
    save_config(config, output_path / "vocoder")
    v_params = sum(v.size for v in vocoder_weights.values())
    print(f"    {len(vocoder_weights)} keys, {v_params:,} params")

    # 7. Text Projections
    print("  [7/7] Text Projections...")
    text_proj_weights = extract_text_projections(all_weights)
    tp_dir = output_path / "text_projections"
    tp_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(tp_dir / "model.safetensors"), text_proj_weights)
    tp_params = sum(v.size for v in text_proj_weights.values())
    print(f"    {len(text_proj_weights)} keys, {tp_params:,} params")

    # Copy upscaler files
    print("\nCopying upscaler files...")
    source_dir = source_path.parent
    is_hf_repo = "/" in source and not Path(source).exists()
    upscaler_files = []

    if is_hf_repo and not cached_only:
        from huggingface_hub import list_repo_files

        upscaler_files = [
            f for f in list_repo_files(source) if UPSCALER_PATTERN.match(f)
        ]
    else:
        upscaler_files = [
            f.name
            for f in source_dir.iterdir()
            if f.is_file() and UPSCALER_PATTERN.match(f.name)
        ]

    if not upscaler_files:
        print("  No upscaler files found")

    for upscaler_file in sorted(upscaler_files):
        dest = output_path / upscaler_file
        if dest.exists():
            print(f"  {upscaler_file}: already exists, skipping")
            continue

        local_candidate = source_dir / upscaler_file
        if local_candidate.is_file():
            shutil.copy2(str(local_candidate), str(dest))
            print(f"  {upscaler_file}: copied")
        elif is_hf_repo and not cached_only:
            from huggingface_hub import hf_hub_download

            print(f"  {upscaler_file}: downloading from {source}...")
            downloaded = hf_hub_download(repo_id=source, filename=upscaler_file)
            shutil.copy2(downloaded, str(dest))
            print(f"  {upscaler_file}: done")
        else:
            print(f"  {upscaler_file}: not found, skipping")

    # Link text_encoder and tokenizer directories
    print("\nLinking text encoder & tokenizer...")
    for subdir in ["text_encoder", "tokenizer"]:
        dest = output_path / subdir
        if dest.exists():
            print(f"  {subdir}/: already exists, skipping")
            continue

        local_candidate = source_dir / subdir
        if local_candidate.is_dir():
            real_path = local_candidate.resolve()
            dest.symlink_to(real_path)
            print(f"  {subdir}/: symlinked to {real_path}")
        elif is_hf_repo and not cached_only:
            from huggingface_hub import list_repo_files, snapshot_download

            repo_files = list_repo_files(source)
            if any(f.startswith(f"{subdir}/") for f in repo_files):
                print(f"  {subdir}/: downloading from {source}...")
                snapshot_download(
                    repo_id=source,
                    allow_patterns=f"{subdir}/*",
                    local_dir=str(output_path),
                )
                print(f"  {subdir}/: done")
                continue
            # Source repo has no text_encoder — fall through to Gemma cache

        # Try local HF cache for Gemma (covers: local source, cached_only, or no subdir in repo)
        gemma_snapshot = find_cached_hf_snapshot(GEMMA_REPO_ID)
        if gemma_snapshot and (Path(gemma_snapshot) / subdir).is_dir():
            real_path = (Path(gemma_snapshot) / subdir).resolve()
            dest.symlink_to(real_path)
            print(f"  {subdir}/: symlinked to {real_path} (from {GEMMA_REPO_ID} cache)")
        elif gemma_snapshot:
            # Gemma stores tokenizer files and model files at snapshot root, no subdirs
            dest.symlink_to(Path(gemma_snapshot).resolve())
            print(f"  {subdir}/: symlinked to {gemma_snapshot} (from {GEMMA_REPO_ID} cache)")
        else:
            print(f"  {subdir}/: not found, skipping (will be resolved at runtime)")

    # Summary
    all_converted = (
        len(transformer_weights)
        + len(vae_decoder_weights)
        + len(vae_encoder_weights)
        + len(audio_decoder_weights)
        + len(audio_encoder_weights)
        + len(vocoder_weights)
        + len(text_proj_weights)
    )
    print(f"\nDone! Converted {all_converted}/{total_keys} keys")
    if all_converted < total_keys:
        known_prefixes = (
            TRANSFORMER_PREFIX,
            VAE_DECODER_PREFIX,
            VAE_ENCODER_PREFIX,
            VAE_STATS_PREFIX,
            AUDIO_DECODER_PREFIX,
            AUDIO_ENCODER_PREFIX,
            AUDIO_STATS_PREFIX,
            VOCODER_PREFIX,
            TEXT_PROJ_PREFIX,
            VIDEO_CONNECTOR_PREFIX,
            AUDIO_CONNECTOR_PREFIX,
        )
        skipped = [
            k for k in all_weights if not any(k.startswith(p) for p in known_prefixes)
        ]
        if skipped:
            print(f"  Skipped {len(skipped)} keys:")
            for k in sorted(skipped)[:20]:
                print(f"    {k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert monolithic LTX-2/2.3 safetensors to modular MLX layout"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="HF repo ID (e.g. Lightricks/LTX-2, Lightricks/LTX-2.3), local directory, or direct safetensors file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for modular layout",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["distilled", "dev"],
        default="distilled",
        help="Model variant (affects VAE decoder config and which file to download)",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        default=False,
        help="Never trigger network downloads — use only locally cached files",
    )
    args = parser.parse_args()

    convert(args.source, Path(args.output), variant=args.variant, cached_only=args.cached_only)
