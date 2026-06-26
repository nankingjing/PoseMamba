#!/usr/bin/env python3
"""Create Awesome List PRs for PoseMamba."""
import os
import subprocess
import tempfile
from pathlib import Path

GITHUB = "nankingjing"
POSE_LINE_AWESOME_MAMBA = (
    "\n[PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional "
    "Global-Local Spatio-Temporal State Space Model](https://arxiv.org/abs/2408.03540) "
    "[[code](https://github.com/nankingjing/PoseMamba)] "
    "[[weights](https://huggingface.co/nankingjings/PoseMamba-weights)] "
    "[[demo](https://huggingface.co/spaces/nankingjings/PoseMamba-Demo)] (AAAI 2025)\n"
)

POSE_PATCH_HPE = """**PoseMamba** (AAAI 2025)

- Bidirectional global-local spatio-temporal SSM
- Purely SSM-based (no convolutions)
- Linear complexity for long sequences
- Spatial reordering strategy
- H36M **38.1 mm** MPJPE (PoseMamba-L)
- 💻 [Code](https://github.com/nankingjing/PoseMamba) · 🤗 [Weights](https://huggingface.co/nankingjings/PoseMamba-weights) · [Demo](https://huggingface.co/spaces/nankingjings/PoseMamba-Demo)
- 📄 [Paper](https://arxiv.org/abs/2408.03540) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32401)
"""


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def pr_awesome_mamba(work: Path):
    repo = "Awesome-Mamba"
    owner = "pengzhangzhi"
    fork = f"{GITHUB}/{repo}"
    run(["gh", "repo", "fork", f"{owner}/{repo}", "--clone=false"], cwd=work)
    dest = work / repo
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    run(["git", "clone", f"https://github.com/{fork}.git", str(dest)])
    run(["git", "remote", "add", "upstream", f"https://github.com/{owner}/{repo}.git"], cwd=dest)
    readme = dest / "README.md"
    text = readme.read_text(encoding="utf-8")
    anchor = "[MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation]"
    if "nankingjing/PoseMamba" in text:
        print("Awesome-Mamba: already has PoseMamba link")
        return
    if anchor not in text:
        raise SystemExit("Anchor not found in Awesome-Mamba README")
    text = text.replace(anchor, POSE_LINE_AWESOME_MAMBA + anchor, 1)
    readme.write_text(text, encoding="utf-8")
    run(["git", "checkout", "-b", "add-posemamba"], cwd=dest)
    run(["git", "add", "README.md"], cwd=dest)
    run(["git", "commit", "-m", "Add PoseMamba (AAAI 2025) with code, weights, and demo links"], cwd=dest)
    run(["git", "push", "-u", "origin", "add-posemamba"], cwd=dest)
    out = subprocess.check_output(
        ["gh", "pr", "create", "--repo", f"{owner}/{repo}", "--head", f"{GITHUB}:add-posemamba",
         "--title", "Add PoseMamba (AAAI 2025) 3D human pose",
         "--body", "Adds PoseMamba with GitHub, Hugging Face weights, and demo space links."],
        cwd=dest, text=True,
    )
    print(out.strip())


def pr_awesome_hpe(work: Path):
    repo = "awesome-human-pose-estimation"
    owner = "umitkacar"
    fork = f"{GITHUB}/{repo}"
    run(["gh", "repo", "fork", f"{owner}/{repo}", "--clone=false"], cwd=work)
    dest = work / repo
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    run(["git", "clone", f"https://github.com/{fork}.git", str(dest)])
    readme = dest / "README.md"
    text = readme.read_text(encoding="utf-8")
    old = """**PoseMamba** (August 2024)

- Bidirectional global-local spatio-temporal SSM
- Purely SSM-based (no convolutions)
- Linear complexity for long sequences
- Spatial reordering strategy
- 📄 [Paper](https://arxiv.org/abs/2408.03540)"""
    if "nankingjing/PoseMamba" in text and "PoseMamba-weights" in text:
        print("awesome-hpe: already updated")
        return
    if old not in text:
        raise SystemExit("PoseMamba block not found in awesome-hpe README")
    text = text.replace(old, POSE_PATCH_HPE, 1)
    readme.write_text(text, encoding="utf-8")
    run(["git", "checkout", "-b", "update-posemamba-links"], cwd=dest)
    run(["git", "add", "README.md"], cwd=dest)
    run(["git", "commit", "-m", "Update PoseMamba entry with code, weights, and AAAI links"], cwd=dest)
    run(["git", "push", "-u", "origin", "update-posemamba-links"], cwd=dest)
    out = subprocess.check_output(
        ["gh", "pr", "create", "--repo", f"{owner}/{repo}", "--head", f"{GITHUB}:update-posemamba-links",
         "--title", "Update PoseMamba with official code and HF resources",
         "--body", "Adds GitHub, Hugging Face weights, demo space, and AAAI paper link to PoseMamba."],
        cwd=dest, text=True,
    )
    print(out.strip())


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        pr_awesome_mamba(work)
        pr_awesome_hpe(work)
