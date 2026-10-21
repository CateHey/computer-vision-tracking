# Bunya HPC Guide

Running the production `cutie_composite` pipeline (SAM3 + CUTIE + YOLO pose + `contacts_v2`), its validation sample, and the older centroid pipeline on UQ's Bunya cluster with the staff account.

| | |
|---|---|
| Host | `bunya.rcc.uq.edu.au` (login nodes `bunya1` to `bunya5`) |
| Username | `uqcvaras` (UQ staff account, approved 2026-10-14) |
| Login | UQ password, then approve the Okta Verify push (SSH keys are not used) |
| Slurm account | `a_<group>` string from a group admin, required on every job |
| Hugging Face token | `~/.hf_token` (SAM3 is a gated model, see section 3) |
| Code, conda env, weights | `~/computer-vision-tracking` (home: 50 GB, never purged) |
| Videos, run outputs | `/scratch/user/uqcvaras/computer-vision-tracking` (300 GB, purged after 90 idle days) |
| GPU jobs | `--partition=gpu_cuda --qos=gpu --gres=gpu:N`, N up to 4, walltime up to 7 days |

Every value above lives in [`scripts/bunya_env.sh`](../../scripts/bunya_env.sh). Source it once per login; the sbatch scripts, the salloc helper and the printed download commands all read it.

## Table of Contents

- [1. Account string (do this first)](#1-account-string-do-this-first)
- [2. Logging in from Windows](#2-logging-in-from-windows)
- [3. First-time setup on Bunya](#3-first-time-setup-on-bunya)
- [4. Upload a video (or a 10 s clip)](#4-upload-a-video-or-a-10-s-clip)
- [5. Production pipeline + validation on a clip (recommended)](#5-production-pipeline--validation-on-a-clip-recommended)
- [6. Centroid pipeline as a batch job (multi-GPU)](#6-centroid-pipeline-as-a-batch-job-multi-gpu)
- [7. Run interactively (salloc)](#7-run-interactively-salloc)
- [8. Centroid pipeline as a Slurm array](#8-centroid-pipeline-as-a-slurm-array)
- [9. Download results](#9-download-results)
- [run_parallel.sh reference](#run_parallelsh-reference)
- [Output structure](#output-structure)
- [Monitoring](#monitoring)
- [Slurm commands](#slurm-commands)
- [Recovery (finding the last batch)](#recovery-finding-the-last-batch)
- [Disk quota](#disk-quota)
- [Roboflow tooling on Bunya](#roboflow-tooling-on-bunya)
- [Troubleshooting](#troubleshooting)

---

## 1. Account string (do this first)

Bunya rejects every job that has no `--account=a_<group>`. RCC does not hand the string out. One of the group's Bunya admins (the list is in the RCC approval email of 14 Oct 2026) has to add `uqcvaras` to the group's Slurm accounting group. Email the admin closest to the lab and ask for exactly that.

Once added, the string shows up on Bunya:

```bash
groups          # look for a_<something>
```

`scripts/bunya_env.sh` picks it up automatically when exactly one `a_*` group is listed and exports it as `SBATCH_ACCOUNT` and `SALLOC_ACCOUNT`, so no script needs editing. To set it by hand:

```bash
export BUNYA_ACCOUNT=a_<group>
source scripts/bunya_env.sh
```

Until the string exists, `salloc` and `sbatch` fail with `Invalid account or account/partition combination specified`. Cloning the repo and uploading videos work without it.

## 2. Logging in from Windows

```powershell
ssh uqcvaras@bunya.rcc.uq.edu.au
# UQ password, then approve the Okta Verify push on your phone (it waits up to 5 min)
```

Optional alias so that `ssh bunya` and `scp file bunya:/path` work. Put this in `C:/Users/uqcvaras/.ssh/config` (create the file if it does not exist):

```
Host bunya
    HostName bunya.rcc.uq.edu.au
    User uqcvaras
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Rules from RCC:

- Every new `ssh` or `scp` asks for password + push again. Windows OpenSSH cannot share connections, so for many files use WinSCP (one login per session) or a single `scp -r` of a folder.
- VS Code Remote-SSH, tunnels and similar IDE connections are not permitted on Bunya. Edit locally, push to GitHub, `git pull` on Bunya.
- Nothing heavy runs on the login nodes (`bunya1` to `bunya5`): no pip or conda installs, no inference. RCC kills those processes.

## 3. First-time setup on Bunya

```bash
# --- on a login node ---
git clone -b develop https://github.com/CateHey/computer-vision-tracking.git ~/computer-vision-tracking   # develop = current pipeline
cd ~/computer-vision-tracking
source scripts/bunya_env.sh        # warns if the account string is still missing

# --- Hugging Face token for SAM3 (facebook/sam3 is gated). Once: accept the terms at
# https://huggingface.co/facebook/sam3, create a read token at https://huggingface.co/settings/tokens
echo hf_xxxxxxxx > ~/.hf_token && chmod 600 ~/.hf_token
source scripts/bunya_env.sh        # now also exports HF_TOKEN

# --- builds must run on a compute node ---
bunya_gpu_shell 1 2                 # 1 GPU for 2 hours
cd ~/computer-vision-tracking && bash scripts/bunya_setup.sh
exit                                # release the node
```

`scripts/bunya_setup.sh` is idempotent and does, in order:

1. Creates `/scratch/user/uqcvaras/computer-vision-tracking/{data/raw,outputs/runs}` and turns `data/raw` and `outputs/runs` in the repo into symlinks pointing there (anything already uploaded to `data/raw` is moved across). Everything else keeps using relative paths; `outputs/slurm` (job logs) stays in home.
2. Loads the `miniforge` module and creates the conda env `rattrack` (Python 3.11 + ffmpeg) under `~/.conda`.
3. Installs PyTorch (CUDA 12.6 wheels, driver >= 525; `TORCH_INDEX=https://download.pytorch.org/whl/cu130` for a newer driver), clones SAM2 into `models/sam2/segment-anything-2` (required by `requirements.txt`), installs `requirements.txt`, `transformers>=5` (SAM3 lives there) and CUTIE into `models/cutie` (falling back to an inference-only dependency set when the full install fails to build).
4. Downloads the weights: SAM2.1 checkpoints, CUTIE (`models/cutie/weights/`) and SAM3 from Hugging Face into `~/.cache/huggingface`. Without `~/.hf_token` the SAM3 step is skipped with a warning and the first job downloads it instead.
5. Imports everything the pipeline, the Excel report and the validation script need, and prints torch, CUDA, transformers and ultralytics versions. `cuda=True` is the line to look for.

The YOLO weights (`models/yolo/yolo26_v11.pt`) come with the git clone.

Re-run the script any time; pieces that exist are skipped. If the miniforge module version changed (`module avail miniforge`), export `BUNYA_CONDA_MODULE=miniforge/<version>` before running.

The interactive run above is optional: the first `sbatch slurm/run_cutie.sbatch` (section 5) runs this setup by itself when the conda env is missing, so from zero the only interactive steps are the clone, the token and the upload.

## 4. Upload a video (or a 10 s clip)

From PowerShell (forward slashes work):

```powershell
scp C:/Users/uqcvaras/Downloads/computer-vision-tracking/data/raw/original.mp4 uqcvaras@bunya.rcc.uq.edu.au:computer-vision-tracking/data/raw/
```

The target is `data/raw` inside the clone (relative to home), which works before setup (plain folder) and after it (symlink into scratch; setup moves anything already there). Scratch files untouched for 90 days are deleted, so keep the master copy on your laptop or RDM. To test on the first seconds of a long video there is no need to cut it: the third argument of `run_cutie.sbatch` limits the run (section 5).

A 10 s test clip (seconds 5 to 15 of the original) is cut on the laptop with ffmpeg and uploaded the same way. Re-encoding makes the cut frame-accurate; a 10 s clip takes seconds:

```powershell
cd C:/Users/uqcvaras/Downloads/computer-vision-tracking
ffmpeg -ss 5 -i <path to the original video> -t 10 -c:v libx264 -crf 18 -pix_fmt yuv420p -an data/clips/clip_5-15s.mp4
ffprobe -v error -show_entries format=duration -of default=nw=1 data/clips/clip_5-15s.mp4   # duration=10.000000
scp data/clips/clip_5-15s.mp4 uqcvaras@bunya.rcc.uq.edu.au:computer-vision-tracking/data/raw/
```

(`-c copy` instead of `-c:v libx264 ...` is faster but starts on the previous keyframe, so the clip may begin before second 5.)

The same cut on Bunya, from a video already on scratch (OpenCV, fine on a login node for a short clip):

```bash
cd ~/computer-vision-tracking && source scripts/bunya_env.sh && bunya_activate
python scripts/trim_video.py data/raw/original.mp4 --start 5 --duration 10 --output data/raw/clip_5-15s.mp4
```

## 5. Production pipeline + validation on a clip (recommended)

`cutie_composite` (SAM3 bootstrap on the first 125 frames, CUTIE propagation, YOLO pose on per-animal composites, `contacts_v2`) is the production pipeline. It runs as one process on one GPU and has no chunk mode. `slurm/run_cutie.sbatch` does the whole thing in one job: installs the environment if it is missing (first run, 20-30 min), runs the pipeline, draws the validation sample (`scripts/validate_contacts.py sample`) from the same run directory, zips the run and prints the download line.

Prerequisites: the token in `~/.hf_token` (section 3) and the video in `data/raw` (section 4).

```bash
cd ~/computer-vision-tracking && source scripts/bunya_env.sh     # account string + ~/.hf_token
sbatch slurm/run_cutie.sbatch data/raw/original.mp4 2 10         # video, number of animals, first N seconds
squeue -u $USER                                                  # PD = waiting for a GPU, R = running
tail -f outputs/slurm/cutie_<JOBID>.out
```

The second argument becomes `detection.max_animals`: the shipped config says 6, the two-rat videos need 2. The third limits the run to the first N seconds (`scan.max_frames`, computed from the video's fps); omit it for the whole video. Knobs: `CONFIG=...`, `OVERRIDES="sam3.sam3_phase_frames=50"`, `VALIDATE=0` (skip the sample), `PER_TYPE=15` (events per contact type), `ZIP=0` (no archive), `SETUP=1` (re-run the setup). A full session needs more memory and time because every frame stays in RAM: `sbatch --mem=128G --time=24:00:00 slurm/run_cutie.sbatch data/raw/original.mp4 2`.

Ten seconds take a few minutes once the models are cached; the first job adds the installation and the model downloads (SAM3 is about 3 GB). The job ends with the archive `outputs/runs/<run>.zip` and the PowerShell lines to copy it into the laptop's `outputs/` folder and unpack it there.

Lines to check in the log:

- `Phase 1 complete. Handoff: n/N rats`: if `n < N`, SAM3 never found every animal and the run must be repeated (see Troubleshooting).
- `Frames with keypoint carry-over`: a high percentage means YOLO often missed an animal and the geometry is interpolated.
- `ContactTracker finalized. Total bouts`: 0 means there is nothing to validate.

The run lands in `outputs/runs/<timestamp>_cutie_composite/`:

| Path | Content |
|------|---------|
| `overlays/cutie_composite_<date>.avi` | overlay video, one colour and label per animal |
| `overlays/cutie_unitary_ratN_<date>.avi` | per-animal composite videos (a second animal visible here means the erase step failed) |
| `contacts/` | `results.xlsx` (every table), CSVs, `event_log.txt`, PDF report |
| `validation/` | `review.xlsx`, `index.html`, `clips/`, `thumbs/`, `manifest.json` |
| `config_used.yaml`, `logs/run.log` | provenance |

Validation round trip, the one manual step:

1. Download and unpack the zip with the lines printed at the end of the job log, open `validation/index.html`, watch each clip, and fill `verdict` (`real`, `flicker`, `wrong_type`, `tracking_error`, `unsure`) and, for `wrong_type`, `correct_type` in `validation/review.xlsx`.
2. Upload the filled workbook back and score it (CPU only, seconds, fine on a login node):

```powershell
scp C:/Users/uqcvaras/Downloads/computer-vision-tracking/outputs/<run>/validation/review.xlsx uqcvaras@bunya.rcc.uq.edu.au:computer-vision-tracking/outputs/runs/<run>/validation/
```

```bash
cd ~/computer-vision-tracking && source scripts/bunya_env.sh && bunya_activate
python scripts/validate_contacts.py score outputs/runs/<run>/validation/review.xlsx
```

`score` writes `validation_summary.md`, `validation_summary.json` and `validation_precision.png` next to the workbook: precision per contact type with Wilson intervals, flicker rate by duration bin, and the minimum-score / minimum-duration threshold tables. To redraw the sample with other settings: `python scripts/validate_contacts.py sample outputs/runs/<run> --per-type 20 --types N2N,N2B` (defaults in `configs/validation.yaml`).

## 6. Centroid pipeline as a batch job (multi-GPU)

The older centroid pipeline (SAM2) through `scripts/run_parallel.sh`. Submit from a login node and log out. Slurm runs the job when GPUs are free.

```bash
ssh uqcvaras@bunya.rcc.uq.edu.au
cd ~/computer-vision-tracking && git pull && source scripts/bunya_env.sh

sbatch scripts/run_pipeline.sbatch data/raw/original.mp4                          # 3 GPUs, 3 chunks
sbatch --gres=gpu:4 --cpus-per-task=16 scripts/run_pipeline.sbatch data/raw/original.mp4 4
sbatch --time=02:00:00 scripts/run_pipeline.sbatch data/raw/original_120s.avi     # short test

squeue -u $USER                                   # PD = waiting for GPUs, R = running
tail -f outputs/slurm/pipeline_<JOBID>.out        # live progress bars
```

Knobs: `CONFIG=... OVERRIDES="detection.confidence=0.3" sbatch scripts/run_pipeline.sbatch ...`. The job calls `scripts/run_parallel.sh`, so the result layout and the printed download commands are the same as in interactive mode. Default walltime is 24 h; `--qos=gpu` allows up to 4 GPUs and 7 days.

Single GPU, single process, no chunking:

```bash
OVERRIDES="video_path=data/raw/original_120s.avi contacts.enabled=true" sbatch slurm/run_centroid.sbatch
```

## 7. Run interactively (salloc)

For watching progress live or debugging.

```bash
cd ~/computer-vision-tracking && source scripts/bunya_env.sh
bunya_gpu_shell 1 2                 # 1 GPU, 2 h (cutie_composite); `bunya_gpu_shell 4 6` for the centroid runner

# now on a GPU node (hostname is not bunyaN). srun starts a clean shell, so:
cd ~/computer-vision-tracking && source scripts/bunya_env.sh && bunya_activate
nvidia-smi -L
python -m src.pipelines.cutie_composite.run --config configs/hpc_cutie_composite.yaml video_path=data/raw/clip_5-15s.mp4 detection.max_animals=2
python scripts/validate_contacts.py sample outputs/runs/<timestamp>_cutie_composite --per-type 15
bash scripts/run_parallel.sh data/raw/original.mp4 4              # centroid pipeline, 4 chunks
exit                                # releases the GPUs
```

`bunya_gpu_shell` is the RCC-recommended salloc, spelled out:

```bash
salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=64G --partition=gpu_cuda --qos=gpu --gres=gpu:4 --time=06:00:00 --job-name=cvt-shell --account=$BUNYA_ACCOUNT srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
```

Pick a GPU type with `--gres=gpu:a100:4` or `--gres=gpu:h100:4` (H100s are capped at 4 per user).

## 8. Centroid pipeline as a Slurm array

When 3 or 4 GPUs on one node are slow to get, let Slurm schedule each centroid chunk on its own:

```bash
cd ~/computer-vision-tracking && source scripts/bunya_env.sh
VIDEO=data/raw/original.mp4 sbatch --array=0-3 slurm/run_chunks.sbatch
squeue -u $USER

# when all four tasks are done:
ls -lt outputs/runs | head
python scripts/merge_chunks.py outputs/runs/<timestamp>_centroid_chunk*/ -o outputs/runs/<timestamp>_centroid_merged
```

Each task reads the frame count from the video and processes its share. Chunk run dirs are named `<timestamp>_centroid_chunk<N>`; pass only the ones from the same submission to the merge.

## 9. Download results

`slurm/run_cutie.sbatch` zips its run directory and prints, at the end of `outputs/slurm/cutie_<JOBID>.out`, the `scp` line for the zip, the `Expand-Archive` line to unpack it into the laptop's `outputs/`, and the upload line for the filled `review.xlsx` (section 5).

`run_parallel.sh` (centroid) prints ready-made `scp` lines at the end of the job log (`outputs/slurm/pipeline_<JOBID>.out`). They look like this (PowerShell):

```powershell
scp uqcvaras@bunya.rcc.uq.edu.au:/scratch/user/uqcvaras/computer-vision-tracking/outputs/runs/<batch>/overlays/<name>_merged.avi C:/Users/uqcvaras/Downloads/computer-vision-tracking/outputs/
scp uqcvaras@bunya.rcc.uq.edu.au:/scratch/user/uqcvaras/computer-vision-tracking/outputs/runs/<batch>/contacts_<batch>.tar.gz C:/Users/uqcvaras/Downloads/computer-vision-tracking/outputs/
```

Whole batch folder: add `-r` and point at `outputs/runs/<batch>`. Each `scp` costs one password + push; for many files use WinSCP.

List what is there first: `ls -lt ~/computer-vision-tracking/outputs/runs/`.

---

## `run_parallel.sh` reference

Runs inside an allocation with N GPUs (interactive, or via `scripts/run_pipeline.sbatch`). Centroid pipeline only: `cutie_composite` holds the whole video in memory and has no chunk mode, so it never goes through this runner.

| Arg | Description | Default |
|-----|-------------|---------|
| `$1` | Video path | (required) |
| `$2` | Number of chunks = GPUs | `3` (clamped to the GPUs visible) |
| `$3` | Config file | `configs/hpc_centroid.yaml` |
| `$4` | Extra overrides (space-separated `key=value`) | none |
| `$5` | Pipeline name, resolves to `src.pipelines.<name>.run` | `centroid` |

For the centroid pipeline it adds `contacts.enabled=true` by itself.

```bash
bash scripts/run_parallel.sh data/raw/original.mp4                 # 3 chunks, centroid
bash scripts/run_parallel.sh data/raw/original.mp4 4               # 4 chunks
bash scripts/run_parallel.sh data/raw/original.mp4 4 configs/hpc_centroid.yaml "detection.confidence=0.3"
bash scripts/run_parallel.sh data/raw/multirat.avi 2 configs/hpc_isolated_composite.yaml "" isolated_composite
```

What it does:

1. Counts the GPUs with `nvidia-smi -L` and clamps the chunk count to it.
2. Creates `outputs/runs/<timestamp>_centroid_batch/` (on scratch through the symlink).
3. Reads total frames, FPS and resolution with OpenCV and splits the frames evenly.
4. Launches one `python -m src.pipelines.centroid.run` per chunk with `CUDA_VISIBLE_DEVICES=<i>`.
5. Shows a live progress bar per chunk with an ETA (parsed from the chunk logs).
6. Finds each chunk's run directory from its log, then runs `scripts/merge_chunks.py` (ffmpeg stream copy for the video, frame-offset merge for the contact CSVs).
7. Compresses `contacts/` into `contacts_<batch>.tar.gz`.
8. Prints the `scp` download commands using `BUNYA_USER`, `BUNYA_HOST` and `BUNYA_LOCAL_DIR` from `scripts/bunya_env.sh`.

## Output structure

```
outputs/runs/<timestamp>_centroid_batch/      (on scratch, via the outputs/runs symlink)
├── chunks/
│   ├── <timestamp>_centroid_chunk0/
│   │   ├── config_used.yaml
│   │   ├── overlays/centroid_<model>_<date>.avi
│   │   ├── contacts/   contacts_per_frame.csv, contact_bouts.csv, session_summary.json
│   │   └── logs/run.log
│   ├── <timestamp>_centroid_chunk1/
│   └── <timestamp>_centroid_chunk2/
├── overlays/
│   └── centroid_<model>_<date>_merged.avi    <- final video
├── contacts/
│   ├── contacts_per_frame.csv                <- all frames combined
│   ├── contact_bouts.csv                     <- all bouts combined
│   ├── session_summary.json                  <- aggregated totals
│   └── (post-processed files and reports/ when the post-processing step is enabled)
├── contacts_<timestamp>_centroid_batch.tar.gz
├── config_used.yaml
└── logs/
    ├── chunk_0.log  chunk_1.log  chunk_2.log
    └── run.log                               <- all chunk logs concatenated
```

See `docs/contacts/` for the contact file formats.

## Monitoring

### Batch job

```bash
squeue -u $USER                                   # state, node, elapsed
tail -f outputs/slurm/pipeline_<JOBID>.out        # the run_parallel.sh progress display
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,AllocTRES%40   # after it finished
```

### Interactive `run_parallel.sh`

If nothing appears right after launching, the script is reading the frame count. Wait a moment.

```bash
LAST=$(ls -dt outputs/runs/*_batch/ 2>/dev/null | head -1); echo "$LAST"
ls -la "$LAST/logs/"
tail -f "$LAST/logs/chunk_0.log"
for f in "$LAST"/logs/chunk_*.log; do echo "=== $(basename "$f") ==="; tail -3 "$f"; done
nvidia-smi                                        # GPU utilisation
ps aux | grep "src.pipelines" | grep -v grep
```

## Slurm commands

```bash
squeue -u $USER                    # my jobs (PD pending, R running)
squeue -u $USER --start            # estimated start time of pending jobs
scancel <JOB_ID>                   # cancel one job (or an array task: scancel <JOB_ID>_<task>)
sacct -u $USER -S today            # history for today
sinfo -p gpu_cuda                  # partition state
rquota                             # home and scratch quotas
module avail miniforge             # conda module versions
nvidia-smi -L                      # GPUs in the current allocation
hostname                           # bunyaN = login node, anything else = compute node
```

## Recovery (finding the last batch)

If a session dropped or you need previous results:

```bash
cd ~/computer-vision-tracking && source scripts/bunya_env.sh && bunya_activate
LAST=$(ls -dt outputs/runs/*_batch/ 2>/dev/null | head -1); echo "Batch: $LAST"

ls "$LAST/overlays/" "$LAST/contacts/" 2>/dev/null || echo "not merged yet"
for f in "$LAST"/logs/chunk_*.log; do echo "=== $(basename "$f") ==="; tail -3 "$f"; done

# merge by hand if every chunk finished but the merge did not run
python scripts/merge_chunks.py "$LAST"/chunks/*_chunk*/ -o "$LAST"

# verify the merged video
python -c "import cv2,glob; v=glob.glob('$LAST/overlays/*_merged.avi')[0]; c=cv2.VideoCapture(v); n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); print(v, n, 'frames', round(n/c.get(cv2.CAP_PROP_FPS),1), 's')"

# re-create the contacts tarball
tar -czf "$LAST/contacts_$(basename "$LAST").tar.gz" -C "$LAST" contacts/
```

A batch job that hit its walltime is simply resubmitted with a longer `--time`; chunk outputs of the aborted run can be deleted.

## Disk quota

```bash
rquota                                              # the numbers that matter
du -sh ~/.conda ~/.cache 2>/dev/null                # home: conda env (~8 GB) + caches
du -sh outputs/runs/* 2>/dev/null | sort -h | tail  # scratch: old batches
conda clean -a -y                                   # drop the conda package cache
rm -rf outputs/runs/<old batch>                     # keep the merged video + contacts tar, drop the rest
```

Home is 50 GB and is never increased: conda env, SAM2 checkpoints and the repo fit comfortably as long as no videos or run outputs land there. Scratch is 300 GB and purges files idle for 90 days, which is why `data/raw` and `outputs/runs` point there.

Full reset of the Bunya side (keeps nothing in home):

```bash
rm -rf ~/computer-vision-tracking ~/.conda/envs/rattrack
# then repeat section 3; videos on scratch are untouched
```

## Roboflow tooling on Bunya

`roboflow/run_extract.sh` (frame extraction for labelling) and `roboflow/train_yolo_pose.sbatch` (YOLO pose retraining) use the same `bunya_env.sh` setup; see `roboflow/TRAINING_GUIDE.md`. The sbatch files of the removed pipelines and the debug runners were deleted in the October 2026 cleanup; git history keeps them.

```bash
bash roboflow/run_extract.sh data/raw/original.mp4 2     # inside bunya_gpu_shell 2 2
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Invalid account or account/partition combination specified` | No account string yet, or a wrong one | Section 1: `groups`, then `export BUNYA_ACCOUNT=a_...` |
| `bunya_env: WARNING: no account string yet` | Same | Same |
| Okta push never arrives | Okta Verify not enrolled, or notifications off | Fix Okta Verify in the UQ Okta portal; the push waits 5 min |
| `Permission denied (publickey,keyboard-interactive)` | Wrong password, or ssh was told to use keys only | Plain `ssh uqcvaras@bunya.rcc.uq.edu.au` and type the UQ password |
| `sbatch: error: Unable to open ... outputs/slurm/...` | Submitted from the wrong directory, or `outputs/slurm` missing | `cd ~/computer-vision-tracking && source scripts/bunya_env.sh` |
| `module: command not found` | Non-login shell | `bash -l`, or `source scripts/bunya_env.sh` (it loads Lmod) |
| `conda env 'rattrack' not found` | Setup never ran | Section 3 |
| `Using device: cpu` or `cuda=False` | On a login node, or a torch build newer than the driver | Run on a GPU node; compare `nvidia-smi` with the index in `bunya_setup.sh` (`TORCH_INDEX=...` to change it) and re-run setup |
| Job pending for a long time (`Resources`, `Priority`) | 3-4 GPUs on one node are scarce | One GPU (`slurm/run_cutie.sbatch`), 2 GPUs, or the array script (section 8) |
| `ERROR: HF_TOKEN is empty` | No Hugging Face token | `echo hf_... > ~/.hf_token && chmod 600 ~/.hf_token`, then re-source `bunya_env.sh` |
| `401`, `403` or `GatedRepoError` loading `facebook/sam3` | Terms not accepted with the token's account | Accept them at https://huggingface.co/facebook/sam3, re-run `bunya_setup.sh` |
| `KeyError: 'sam3_video'` or `cannot import name 'Sam3VideoModel'` | transformers 4.x | `pip install -U "transformers>=5.0"` inside the env (setup does this) |
| CUTIE install fails building `cchardet`, `netifaces` or `PySide6` | Cutie's GUI/training extras | Setup falls back to `--no-deps` plus the inference packages by itself; re-run it |
| `Phase 1 complete. Handoff: n/N rats` with `n < N` | SAM3 missed an animal in the first frames | Check the animal count passed to `run_cutie.sbatch`; try `OVERRIDES="sam3.score_threshold=0.3"` or another `sam3.sam3_phase_frames` |
| `No events/bouts table with pair_key found` from `validate_contacts.py` | The clip produced no contact events | Nothing to validate: use a longer clip, or check `Total bouts` in the log |
| `QOSMaxGRESPerJob` | More than 4 GPUs requested | `--gres=gpu:4` at most |
| Process killed on a login node | pip, conda or inference on `bunyaN` | `bunya_gpu_shell` |
| `CUDA out of memory` | Several processes on one GPU | One chunk per GPU (the default) |
| `git pull` conflicts on Bunya | Edits made on Bunya | `git stash && git pull && git stash pop`, or discard with `git checkout -- .` |
| `git status` shows `data/raw` | Setup never ran, or `.git/info/exclude` was lost | Re-run `bash scripts/bunya_setup.sh` |
| Files vanished from `data/raw` or `outputs/runs` | 90-day scratch purge | Re-upload; keep master copies locally |
| Exit 1 on a chunk | Python error | `logs/chunk_N.log` in the batch dir |
