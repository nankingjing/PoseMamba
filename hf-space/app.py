"""PoseMamba Hugging Face Space — demo gallery + links to full GPU inference."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import gradio as gr

GITHUB = "https://github.com/nankingjing/PoseMamba"
WEIGHTS = "https://huggingface.co/nankingjings/PoseMamba-weights"
PAPER_AAAI = "https://ojs.aaai.org/index.php/AAAI/article/view/32401"
PAPER_ARXIV = "https://arxiv.org/abs/2408.03540"
DISCUSSION = "https://github.com/nankingjing/PoseMamba/discussions/21"
COLAB = "https://colab.research.google.com/github/nankingjing/PoseMamba/blob/main/notebooks/PoseMamba_Demo.ipynb"

ASSETS = {
    "sample_gif": "https://github.com/nankingjing/PoseMamba/raw/main/sample_video.gif",
    "sample_mp4": "https://github.com/nankingjing/PoseMamba/raw/main/sample_video.mp4",
}

CACHE = Path(__file__).parent / ".cache"


def _fetch(url: str, name: str) -> str:
    CACHE.mkdir(exist_ok=True)
    dest = CACHE / name
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return str(dest)


INTRO = f"""
# PoseMamba Demo (AAAI 2025)

**PoseMamba-L**: **38.1 mm** MPJPE on Human3.6M (detected 2D) · **6.7M** params · **27.9G** MACs

Bidirectional spatio-temporal **Mamba/SSM** for monocular **3D human pose** (2D→3D lifting).

| Resource | Link |
|----------|------|
| Code | [{GITHUB}]({GITHUB}) |
| Weights | [{WEIGHTS}]({WEIGHTS}) |
| Paper (AAAI) | [link]({PAPER_AAAI}) |
| arXiv | [2408.03540]({PAPER_ARXIV}) |
| Install FAQ | [Discussion #21]({DISCUSSION}) |

> Full video inference needs **CUDA + selective_scan** build. Use **[Colab notebook]({COLAB})** or local `vis.py` for your own videos.
"""


def load_gif():
    return _fetch(ASSETS["sample_gif"], "sample_video.gif")


def load_mp4():
    try:
        return _fetch(ASSETS["sample_mp4"], "sample_video.mp4")
    except Exception:
        return None


with gr.Blocks(title="PoseMamba Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### In-the-wild demo (official)")
            gif_out = gr.Image(label="sample_video.gif", type="filepath")
            btn_gif = gr.Button("Load demo GIF", variant="primary")
        with gr.Column():
            gr.Markdown("### Sample video")
            vid_out = gr.Video(label="sample_video.mp4")
            btn_vid = gr.Button("Load sample MP4")
    gr.Markdown(
        f"""
### Run on your own video

1. **[Open Colab notebook]({COLAB})** (recommended, free GPU)
2. Or clone the repo and run: `python vis.py --video your.mp4 --gpu 0`
3. Download weights from **[Hugging Face]({WEIGHTS})**

See **[Installation & FAQ]({DISCUSSION})** for `selective_scan` build issues.
"""
    )
    btn_gif.click(load_gif, outputs=gif_out)
    btn_vid.click(load_mp4, outputs=vid_out)
    demo.load(load_gif, outputs=gif_out)

if __name__ == "__main__":
    demo.launch()
