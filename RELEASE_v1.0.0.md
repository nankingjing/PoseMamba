# PoseMamba v1.0.0

**AAAI 2025** official release — documentation, issue templates, and pretrained weight links.

## Highlights

- PoseMamba-L: **38.1 mm** MPJPE on Human3.6M (P1, detected 2D), **6.7M** params, **27.9G** MACs
- Bidirectional spatio-temporal **Mamba/SSM** for monocular 3D human pose estimation
- In-the-wild demo via `vis.py`

## What's in this release

- Rewritten README (SOTA table, Quick Start, FAQ)
- [Installation & FAQ Discussion](https://github.com/nankingjing/PoseMamba/discussions/21) (pinned)
- Issue templates (Bug Report / Question)
- Fixed config paths: `configs/pose3d/`
- Correct evaluation: `--evaluate best_epoch.bin` (see `eval.sh`)

## Pretrained weights

| Model | Params | Download |
|-------|--------|----------|
| PoseMamba-S | 0.9M | [Google Drive](https://drive.google.com/file/d/1LZtEjeiAIx6LXFmjoyKKzbaCPV3R1-P7/view?usp=sharing) |
| PoseMamba-B | 3.4M | [Google Drive](https://drive.google.com/file/d/1aP6WAq5fKNIqyYcI_ZnYbuagR3_zVik2/view?usp=sharing) |
| PoseMamba-L | 6.7M | [Google Drive](https://drive.google.com/file/d/16_Tg0Aqzgih243_dflyFv0UB79gU9u8q/view?usp=sharing) |
| All-in-one | — | [Bundle](https://drive.google.com/file/d/1WFRAeal8W6ntrTPNrf-SNywdgupj0-S8/view?usp=sharing) |

> Hugging Face mirrors coming soon. Track progress in repo README.

## Quick start

```bash
conda create -n posemamba python=3.8.5 && conda activate posemamba
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
cd kernels/selective_scan && pip install -e . && cd ../..
python vis.py --video sample_video.mp4 --gpu 0
```

## Paper

- [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32401)
- [arXiv:2408.03540](https://arxiv.org/abs/2408.03540)

## Citation

```bibtex
@inproceedings{huang2025posemamba,
  title={PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model},
  author={Huang, Yunlong and Liu, Junshuo and Xian, Ke and Qiu, Robert Caiming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
