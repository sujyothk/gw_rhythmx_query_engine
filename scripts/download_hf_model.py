from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    p = argparse.ArgumentParser(description="Download a Hugging Face model to a local folder (for offline use).")
    p.add_argument("--model-id", required=True, help="e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--out-dir", required=True, help="Local directory to store model files")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # snapshot_download returns the cache location; local_dir ensures a clean folder for submission
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.model_id} to: {out_dir}")


if __name__ == "__main__":
    main()
