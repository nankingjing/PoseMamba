# 知乎推文草稿 · PoseMamba

**标题（选一）：**
1. 【AAAI 2025 一作开源】PoseMamba：用 Mamba 做 3D 人体姿态，比 MotionBERT 省 84% 算力
2. 线性复杂度 3D 姿态估计 PoseMamba 开源了，Hugging Face 可直接下权重

---

## 正文

最近在 AAAI 2025 发表了 **PoseMamba**——用 **双向时空 State Space Model (Mamba)** 做单目视频 **3D 人体姿态估计**（2D→3D lifting）。

### 为什么值得关注？

- **PoseMamba-L**：Human3.6M **38.1 mm** MPJPE（P1，detected 2D）
- 仅 **6.7M** 参数 / **27.9G** MACs，比 MotionBERT **省约 84% 计算量**
- 比 MotionAGFormer **快 2.8×**，显存降 **64.7%**
- **纯 SSM 架构**，双向全局-局部时空建模

### 开源资源（已整理）

- 代码：https://github.com/nankingjing/PoseMamba
- 预训练权重（HF）：https://huggingface.co/nankingjings/PoseMamba-weights
- **在线 Demo**：https://huggingface.co/spaces/nankingjings/PoseMamba-Demo
- **Colab**：https://colab.research.google.com/github/nankingjing/PoseMamba/blob/main/notebooks/PoseMamba_Demo.ipynb
- 安装排错：https://github.com/nankingjing/PoseMamba/discussions/21

### 快速试玩

```bash
git clone https://github.com/nankingjing/PoseMamba
cd PoseMamba
# 按 README 装环境 + kernels/selective_scan
python vis.py --video sample_video.mp4 --gpu 0
```

也可以用仓库里的 **Colab notebook** 一键跑 wild video demo。

### 论文

- AAAI 2025：https://ojs.aaai.org/index.php/AAAI/article/view/32401
- arXiv：https://arxiv.org/abs/2408.03540

欢迎 star / issue，有问题先看 pinned FAQ discussion。

#3D人体姿态估计 #Mamba #AAAI2025 #开源 #计算机视觉
