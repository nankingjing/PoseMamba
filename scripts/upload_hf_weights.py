#!/usr/bin/env python3
"""
Upload PoseMamba pretrained weights to Hugging Face Hub.

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   # or set HF_TOKEN

Usage:
  # After downloading Google Drive weights locally:
  python scripts/upload_hf_weights.py --weights-dir ./checkpoints_hf --repo nankingjing/PoseMamba-weights

Expected local layout:
  checkpoints_hf/
    PoseMamba-S.pth.tr
    PoseMamba-B.pth.tr
    PoseMamba-L.pth.tr
"""
from __future__ import annotations

import argparse
from pathlib import Path

MODEL_CARD = """---
license: apache-2.0
tags:
  - pose-estimation
  - mamba
  - 3d-pose
  - aaai2025
  - human-pose-estimation
---

# PoseMamba Pretrained Weights

Official pretrained checkpoints for [PoseMamba](https://github.com/nankingjing/PoseMamba) (AAAI 2025).

| Model | Params | File |
|-------|--------|------|
| PoseMamba-S | 0.9M | PoseMamba-S.pth.tr |
| PoseMamba-B | 3.4M | PoseMamba-B.pth.tr |
| PoseMamba-L | 6.7M | PoseMamba-L.pth.tr |

## Usage

```python
import torch
ckpt = torch.load("PoseMamba-L.pth.tr", map_location="cpu")
state = ckpt["model_pos"]
```

See the [GitHub repo](https://github.com/nankingjing/PoseMamba) for training, evaluation (`eval.sh`), and demo (`vis.py`).

## Citation

```bibtex
@inproceedings{huang2025posemamba,
  title={PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model},
  author={Huang, Yunlong and Liu, Junshuo and Xian, Ke and Qiu, Robert Caiming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--repo", default="nankingjings/PoseMamba-weights")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise SystemExit("Install: pip install huggingface_hub")

    weights_dir = args.weights_dir
    if not weights_dir.is_dir():
        raise SystemExit(f"Directory not found: {weights_dir}")

    files = list(weights_dir.glob("*.pth.tr")) + list(weights_dir.glob("*.bin"))
    if not files:
        raise SystemExit(f"No .pth.tr or .bin files in {weights_dir}")

    api = HfApi()
    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    for f in files:
        print(f"Uploading {f.name}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=args.repo,
            repo_type="model",
        )

    readme_path = weights_dir / "README.md"
    readme_path.write_text(MODEL_CARD, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
