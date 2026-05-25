<p align="center">
  <h1 align="center"> CRAG: Can 3D Generative Models Help 3D Assembly? </h1>
  <h3 align="center">
    <a href="https://icml.cc/" target="_blank" rel="noopener noreferrer">
      ICML 2026
    </a>
  </h3>

  <p align="center">
  A unified framework that couples 3D assembly with 3D generation: assembly provides part-level structural priors for generation, while generation injects holistic shape context that resolves ambiguities in assembly — enabling pose prediction <i>and</i> missing-geometry synthesis in a single model.
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2602.22629" target="_blank"><img src="https://img.shields.io/badge/arXiv-2602.22629-b31b1b" alt="arXiv"></a>
    <a href="https://icml.cc/" target="_blank">
      <img src="https://img.shields.io/badge/ICML-2026-4b96dc" alt="ICML 2026">
    </a>
  </p>
  <p align="center">
    <a href="https://github.com/JDScript">Zeyu Jiang</a>
    ·
    <a href="https://scholar.google.com/citations?user=90IoeJsAAAAJ">Sihang Li</a>
    ·
    <a href="https://github.com/kevintsq">Siqi Tan</a>
    ·
    <a href="https://chenyang-joe.github.io">Chenyang Xu</a>
    ·
    Juexiao Zhang
    ·
    Julia Galway-Witham
    ·
    Xue Wang
    ·
    Scott A. Williams
    ·
    <a href="https://scholar.google.com/citations?user=JqLHsvYAAAAJ&hl=en">Radu Iovita</a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=YeG8ZM0AAAAJ">Chen Feng✉</a>
    ·
    <a href="https://jingz6676.github.io/">Jing Zhang✉</a>
  </p>
  <p align="center">
    ✉ Corresponding author
  </p>

  <div align="center"></div>

## 🔊 News & Todos
- `2026/05/25`: We release the CRAG codebase. Pretrained checkpoints and data will be released shortly — stay tuned!
- `2026/05/01`: CRAG has been accepted to **ICML 2026** 🎉

**Todos**
- [ ] Release pretrained checkpoints (Stage 1 & Stage 2, Breaking Bad, PartNeXt and Omni variants)

## 📖 Table of Contents

- [📄 Documentation](#-documentation)
  - [⏩ Installation](#-installation)
  - [💾 Data Preparation](#-data-preparation)
  - [🎯 Evaluation](#-evaluation)
  - [🎮 Training](#-training)
- [🙋 FAQs](#-faqs)
- [📚 Citation](#-citation)
- [📝 License](#-license)
- [🙏 Acknowledgement](#-acknowledgement)

## 📄 Documentation

### ⏩ **Installation**
We recommend using [uv](https://docs.astral.sh/uv/) to manage the dependencies. Follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/) to install uv. Then simply run

```bash
uv sync
source .venv/bin/activate
```

The project is pinned to Python `3.12.3` and PyTorch `2.8.0 + CUDA 12.8`.

If you run into issues, try removing the `.venv` directory and running `uv clean && uv self-update` before re-installing.

### 💾 **Data Preparation**
CRAG trains and evaluates across multiple assembly datasets. All datasets are stored in HDF5 format aligned to [GARF](https://github.com/ai4ce/GARF) with additional schema fields for renderings. We provide our processed datasets below:

<table>
  <tr>
    <th>Dataset</th>
    <th>Source</th>
    <th>Download</th>
  </tr>
  <tr>
    <td>Breaking Bad</td>
    <td><a href="https://breaking-bad-dataset.github.io/" target="_blank">Project Page</a></td>
    <td><a href="https://jdscript-my.sharepoint.com/:u:/g/personal/shared_jdscript_app/IQASZITu4dKFTpCTVfxusL6eAT-cE4vcMKgDoFumgGtgYu8?e=fNzQEO" target="_blank">OneDrive</a></td>
  </tr>
  <tr>
    <td>PartNeXt</td>
    <td><a href=https://authoritywang.github.io/partnext/" target="_blank">Project Page</a></td>
    <td><a href="https://jdscript-my.sharepoint.com/:u:/g/personal/shared_jdscript_app/IQDTXiYjM3AsQZO-VYZFyAEiAeg-TlAo9B958ae1NNUaw-A?e=zvXXNR" target="_blank">OneDrive</a></td>
  </tr>
</table>


### 🎯 **Evaluation**
We provide an evaluation script in `scripts/eval.sh`, modify the `experiment` and `ckpt_path` arguments to point to your desired checkpoint and config.

Results (per-sample transformations, generated results and aggregate metrics) are written under `logs/<EXPERIMENT_NAME>/`. For the most reliable per-sample dumps we recommend single-GPU evaluation.

### 🎮 **Training**
Training proceeds in two stages: an assembly pre-training stage, followed by a generation-coupled fine-tuning stage. We provide one launcher per dataset family.

#### ⭐ **Stage 1: Assembly Pre-training**

Breaking Bad + MorphoSource + Skull:
```bash
bash scripts/train_bb.sh stg_1_bb
```

PartNeXt:
```bash
bash scripts/train_partnext.sh stg_1_partnext
```

Both scripts auto-resume from `output/<EXPERIMENT_NAME>/last.ckpt` if it exists, copy the HDF5 files to `/dev/shm` for faster I/O, and launch via `torchrun` using the `PET_*` environment variables provided by your distributed launcher.

#### ⭐ **Stage 2: Generation-Coupled Fine-tuning**
Stage 2 jointly fine-tunes the assembly backbone together with the generative branch (a TripoSG-based shape head):

```bash
bash scripts/train_bb.sh stg_2_bb_finetune
# or
bash scripts/train_partnext.sh stg_2_partnext_finetune
```

You will typically want to point `ckpt_path` at the Stage 1 checkpoint the first time you launch Stage 2. The launcher's auto-resume logic will then take over for subsequent runs.

## 🙋 FAQs
For frequently asked questions, please refer to our [GitHub Issues](https://github.com/ai4ce/CRAG/issues) page. Search existing threads, or open a new issue if your question hasn't been answered yet.

## 📚 Citation
If you find this project useful, please consider citing our paper:

```bibtex
@inproceedings{jiang2026crag,
  title     = {CRAG: Can 3D Generative Models Help 3D Assembly?},
  author    = {Jiang, Zeyu and Li, Sihang and Tan, Siqi and Xu, Chenyang and Zhang, Juexiao and Galway-Witham, Julia and Wang, Xue and Williams, Scott A. and Iovita, Radu and Feng, Chen and Zhang, Jing},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

Our codebase builds on the excellent work of [GARF](https://github.com/ai4ce/GARF) and [TripoSG](https://github.com/VAST-AI-Research/TripoSG).

## 📝 License
This project is licensed under the GPL License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgement
We thank the authors of the [Breaking Bad](https://breaking-bad-dataset.github.io/) and [PartNeXt](https://github.com/AuthorityWang/PartNeXt) datasets for releasing the data that underpins our pre-training and evaluation, and the [GARF](https://github.com/ai4ce/GARF) and [TripoSG](https://github.com/VAST-AI-Research/TripoSG) projects whose codebases CRAG is built on top of.

