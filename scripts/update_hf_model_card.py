#!/usr/bin/env python3
"""Update Hugging Face model card for PoseMamba-weights."""
from pathlib import Path
from huggingface_hub import HfApi

REPO = "nankingjings/PoseMamba-weights"
README = """---
license: apache-2.0
tags:
  - pose-estimation
  - mamba
  - ssm
  - 3d-pose
  - aaai2025
  - human-pose-estimation
  - pytorch
arxiv: 2408.03540
datasets:
  - human3.6m
language:
  - en
pipeline_tag: other
---

# PoseMamba Pretrained Weights

Official checkpoints for **[PoseMamba](https://github.com/nankingjing/PoseMamba)** (AAAI 2025).

> Paper: [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32401) · [arXiv:2408.03540](https://arxiv.org/abs/2408.03540)

## Models

| Model | File | Params | MACs | H36M P1 (2D det) |
|-------|------|--------|------|------------------|
| PoseMamba-S | `PoseMamba_S.bin` | 0.9M | 3.6G | 41.8 mm |
| PoseMamba-B | `PoseMamba_B.bin` | 3.4M | 13.9G | 40.8 mm |
| PoseMamba-L | `PoseMamba_L.bin` | 6.7M | 27.9G | **38.1 mm** |

## Load checkpoint

```python
import torch
ckpt = torch.load("PoseMamba_L.bin", map_location="cpu")
state_dict = ckpt["model_pos"]
```

## Links

- **Code**: https://github.com/nankingjing/PoseMamba
- **Demo Space**: https://huggingface.co/spaces/nankingjings/PoseMamba-Demo
- **Colab**: https://colab.research.google.com/github/nankingjing/PoseMamba/blob/main/notebooks/PoseMamba_Demo.ipynb
- **FAQ**: https://github.com/nankingjing/PoseMamba/discussions/21

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

if __name__ == "__main__":
    from huggingface_hub import login
    login(add_to_git_credential=False)
    api = HfApi()
    path = Path(__file__).resolve().parents[1] / "checkpoints_hf" / "README.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(README, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md", repo_id=REPO, repo_type="model")
    print(f"Updated https://huggingface.co/{REPO}")
