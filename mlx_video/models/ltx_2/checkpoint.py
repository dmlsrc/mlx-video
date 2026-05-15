"""LTX checkpoint resolution, header reading, and HF Hub cache utilities."""

import json
import re
import struct
from pathlib import Path
from typing import Optional


# Matches monolithic model files: ltx-2-19b-distilled.safetensors, ltx-2.3-22b-dev.safetensors, etc.
MONOLITHIC_PATTERN = re.compile(
    r"^ltx-[\d.]+-\d+b-(?P<variant>distilled|dev)(?:-[\d.]+)?\.safetensors$"
)

# Matches upscaler files like ltx-2-spatial-upscaler-x2-1.0.safetensors, etc.
UPSCALER_PATTERN = re.compile(
    r"^ltx-[\d.]+-(?:spatial|temporal)-upscaler-.+\.safetensors$"
)


def find_cached_hf_snapshot(repo_id: str, required_file: str = "config.json") -> Optional[str]:
    """Return the local HF hub snapshot path for a repo, or None if not cached.

    Never triggers a download — local_files_only=True.
    """
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id, local_files_only=True)
        if (Path(path) / required_file).exists():
            return path
    except Exception:
        pass
    return None


def version_key(name) -> tuple:
    """Extract version tuple from a safetensors filename for newest-first sorting."""
    m = re.search(r"-(\d+(?:\.\d+)+)\.safetensors$", str(name))
    if m:
        return tuple(int(x) for x in m.group(1).split("."))
    return (0,)


def resolve_source(source: str, variant: str, local_files_only: bool = False) -> Path:
    """Resolve source to a monolithic safetensors file path.

    Args:
        source: HF repo ID (e.g. "Lightricks/LTX-2"), local directory, or direct file path.
        variant: Model variant ("distilled" or "dev") to select the right file.
        local_files_only: If True, raise instead of downloading when not cached.

    Returns:
        Path to the monolithic safetensors file.
    """
    source_path = Path(source)

    # Direct file path
    if source_path.is_file():
        return source_path

    # Local directory — find the variant's safetensors file
    if source_path.is_dir():
        matches = [
            f for f in sorted(source_path.glob("ltx-*b-*.safetensors"))
            if (m := MONOLITHIC_PATTERN.match(f.name)) and m.group("variant") == variant
        ]
        if matches:
            return sorted(matches, key=version_key, reverse=True)[0]

        # Broader fallback
        all_mono = sorted(source_path.glob("ltx-*.safetensors"))
        for f in all_mono:
            if variant in f.name and MONOLITHIC_PATTERN.match(f.name):
                return f

        raise FileNotFoundError(
            f"No monolithic *-{variant}.safetensors found in {source_path}. "
            f"Files found: {[f.name for f in all_mono]}"
        )

    # HF repo ID — check local cache first, then download
    if "/" in source and not source_path.exists():
        from huggingface_hub import snapshot_download

        try:
            snapshot_path = Path(snapshot_download(source, local_files_only=True))
            matches = [
                f for f in sorted(snapshot_path.glob("ltx-*b-*.safetensors"))
                if (m := MONOLITHIC_PATTERN.match(f.name)) and m.group("variant") == variant
            ]
            if matches:
                return sorted(matches, key=version_key, reverse=True)[0]
        except Exception:
            pass

        if local_files_only:
            raise FileNotFoundError(
                f"{source} is not cached locally. "
                f"Download it first or run without --cached-only."
            )

        from huggingface_hub import hf_hub_download, list_repo_files

        repo_files = list_repo_files(source)
        candidates = [
            f for f in repo_files
            if (m := MONOLITHIC_PATTERN.match(f)) and m.group("variant") == variant
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No *-{variant}.safetensors found in {source}. "
                f"Available: {[f for f in repo_files if f.endswith('.safetensors')]}"
            )

        target = sorted(candidates, key=version_key, reverse=True)[0]
        print(f"Downloading {target} from {source}...")
        return Path(hf_hub_download(repo_id=source, filename=target))

    raise FileNotFoundError(
        f"Source not found: {source}. Provide an HF repo ID, local directory, or file path."
    )


def read_embedded_config(source_path: Path) -> dict:
    """Read the embedded config JSON from a safetensors file header without loading weights."""
    try:
        with open(source_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        config_str = header.get("__metadata__", {}).get("config", "")
        return json.loads(config_str) if config_str else {}
    except Exception:
        return {}
