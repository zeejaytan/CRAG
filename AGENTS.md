# AGENTS.md — CRAG (project)

Follow the workspace root **`../AGENTS.md`** (laptop ↔ GitHub ↔ Spartan) for all shared rules. This file only adds CRAG-specific paths and domain notes.

## CRAG paths

| Role | Value |
|------|--------|
| GitHub fork (`origin`) | `zeejaytan/CRAG` |
| Upstream | `ai4ce/CRAG` |
| Spartan checkout (`REMOTE_ROOT`) | `/data/gpfs/projects/punim2657/CRAG` |
| SSH | `Host spartan`, user `zhuojiat` |
| Remote helpers | `scripts/remote/pull_and_sbatch.sh`, `job_status.sh`, `fetch_artifacts.sh` |

Default branch is **`main`**. Upstream carries a `third_party/tripo_sg` submodule
(`JDScript/TripoSG` fork) — clone with `--recurse-submodules` and run
`git submodule update --init --recursive` after every fresh clone. Heavy data on
Spartan only (gitignored): `data/`, `checkpoints/`, `output/`, `logs/`,
`*.hdf5` / `*.h5`, `*.ckpt`, `pretrained/`. Local rsync landing zone: `artifacts/`.

**Write rules:** new analysis/code → `scripts/`; versioned Slurm → `scripts/hpc/`;
method notes → `docs/notes/`; fetched samples → `artifacts/` (not source); HPC
paths → `CLAUDE.local.md`. Upstream method code lives in `source/` plus
`train.py` / `test.py` (Hydra entry points) — do not refactor it without a ticket.
Do not add files at the CRAG root.

Typical loop:

```bash
git push origin HEAD
./scripts/remote/pull_and_sbatch.sh scripts/hpc/<job>.slurm
./scripts/remote/job_status.sh
./scripts/remote/fetch_artifacts.sh logs/<experiment> ./artifacts/
```

## Domain / debugging

CRAG is a 3D assembly + 3D generation method built on GARF plus a TripoSG shape
head (ICML 2026, `ai4ce/CRAG`). Its HDF5 datasets are GARF-aligned with extra
rendering fields, so GARF-side tooling mostly applies. Training is two-stage:
`scripts/train_bb.sh` / `scripts/train_partnext.sh` (assembly pre-train, then
generation-coupled fine-tune); both expect the `PET_*` distributed env and copy
HDF5s to `/dev/shm`. Evaluation is `test.py` with `experiment=` + `ckpt_path=`
(see `scripts/eval.sh`).

As of 2026-09-16 upstream has **not released pretrained checkpoints or data
files** (README Todos still open) — any Juglet test needs training or waits on
that release. Install is Linux-only (`uv sync`; `pyproject.toml` pins
`sys_platform == 'linux'`, Python 3.12.3, torch 2.8 + cu128), so build on
Spartan, not the laptop. Prefer probes over guesses (Hydra/OmegaConf, dataloader
length, HDF5 splits). Do not invent config keys. Session notes may live under
`docs/notes/`.

## Agent skills

Configured here so this repo works when opened on its own, not only from the `C:\PR`
umbrella. The full text of each convention lives at the workspace root; these are the
parts an agent needs before it can act.

- **Issue tracker — local markdown.** One feature per directory: the spec at
  `.scratch/<feature>/spec.md`, tickets one per file at
  `.scratch/<feature>/issues/<NN>-<slug>.md`, numbered from `01` in dependency order.
  Every ticket carries an **`Answers:`** line naming the question in `intent/` it exists
  to settle -- `R1` for this project, `U6` for the workspace, or `none` for routine
  work. Conventions and the ticket template: `../docs/agents/issue-tracker.md`.
- **Triage labels.** `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
  `wontfix`, recorded as a `Status:` line near the top of the ticket. Details:
  `../docs/agents/triage-labels.md`.
- **Domain docs — single-context.** Three different things, kept apart: **this file** is
  how to work here and the traps; **`CONTEXT.md`** at the repo root is the glossary, and
  `/domain-modeling` creates it lazily when the first term is actually resolved — do not
  create it empty; **`../docs/glossary.md`** is the cross-project measurement vocabulary
  (`part_acc`, chamfer distance, best-of-N) and outranks any local redefinition. ADRs go
  under `docs/adr/`. Details: `../docs/agents/domain.md`.
- **Intent.** [`intent/`](intent/) holds what we are trying to establish and what would
  settle it -- prefix **`R`**, permanent, numbers never reused. `/to-intent` opens a
  question or writes a finished ticket's result back into one. Check the loop is wired
  with `python ../scripts/check_intent_links.py`.

**Do not run `/setup-matt-pocock-skills` in this repo.** It would replace the above with
its own defaults, and its ticket template has no `Answers:` line -- tickets would stop
being connected to the question they exist to answer, silently.
