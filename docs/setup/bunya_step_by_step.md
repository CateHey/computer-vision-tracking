# Bunya step by step (verified commands only)

Setup from zero with the staff account `uqcvaras`, one command at a time.
Only commands that were run on Bunya and worked are recorded here. No helper scripts, no hidden variables.

## 1. Log in

From PowerShell on the laptop. Asks for the UQ password, then an Okta Verify push.

```powershell
ssh uqcvaras@bunya.rcc.uq.edu.au


```

## 2. Clone the repo (develop branch)

If `~/computer-vision-tracking` already exists from an earlier attempt, delete it first.

```bash
rm -rf ~/computer-vision-tracking
git clone --depth 1 -b develop https://github.com/CateHey/computer-vision-tracking.git ~/computer-vision-tracking
cd ~/computer-vision-tracking
git log --oneline -1
```

Update after every change pushed from the laptop:

```bash
cd ~/computer-vision-tracking && git pull
```

## 3. Upload the video

From PowerShell on the laptop, not from Bunya. Example with the 10 s clip in the repo:

```powershell
scp C:\Users\uqcvaras\Downloads\computer-vision-tracking\data\clips\output-10s.mp4 uqcvaras@bunya.rcc.uq.edu.au:computer-vision-tracking/data/raw/original.mp4
```

Check on Bunya:

```bash
ls -lh ~/computer-vision-tracking/data/raw/original.mp4
```

## 4. Load conda

`module load miniforge` without a version fails with "exist but cannot be loaded as requested".
The full version is required. List the available ones with `module spider miniforge`.

```bash
module load miniforge/26.3.2-1
which conda          # -> /sw/local/rocky8/noarch/rcc/software/miniforge/26.1.0-0/bin/conda
```

`conda --version` may print nothing right after loading the module; `which conda` is the
reliable check. The module also defines a helper `conda-activate` that just runs
`eval "$(conda shell.bash hook)"` and activates the base env. The plain hook below does
the same and is what the job scripts use.

```bash
eval "$(conda shell.bash hook)"     # enables "conda activate" in this shell
```

## 5. Create the environment

Runs on the login node (no Slurm account needed). Downloads about 200 MB, takes a few minutes.
The env lands in `~/.conda/envs/rattrack`. A "newer version of conda exists" warning is harmless.

```bash
conda create -y -n rattrack -c conda-forge python=3.11 ffmpeg
conda activate rattrack
python --version     # -> Python 3.11.x
```

## 6. Install the Python packages

**Lesson:** a plain `pip install torch ...` on the login node failed half way with
`OSError: [Errno 122] Disk quota exceeded`, although `rquota` showed home at 1 of 50 GB.
pip unpacks wheels in the node's temporary folder, which is small. Fix: point `TMPDIR`
at scratch (300 GB) and use `--no-cache-dir` so nothing piles up in `~/.cache/pip`.

Check space with one line:

```bash
rquota; du -sh ~/.[!.]* ~/* 2>/dev/null | sort -h | tail
```

Everything in one block (about 10-15 min). It stops at the first failure:

```bash
rat
mkdir -p /scratch/user/uqcvaras/tmp && export TMPDIR=/scratch/user/uqcvaras/tmp
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
grep -v segment-anything requirements.txt > $TMPDIR/req.txt && \
pip install --no-cache-dir -r $TMPDIR/req.txt && \
pip install --no-cache-dir "transformers>=5.0" accelerate && \
{ [ -d models/cutie ] || git clone https://github.com/hkchengrex/Cutie.git models/cutie; } && \
{ pip install --no-cache-dir -e models/cutie || { pip install --no-cache-dir --no-deps -e models/cutie && pip install --no-cache-dir hydra-core omegaconf einops easydict gitpython; }; } && \
python -c "from cutie.utils.download_models import download_models_if_needed; download_models_if_needed()" && \
python -c "import torch, transformers, ultralytics, cutie; from transformers import Sam3VideoModel; print('ALL OK', torch.__version__, transformers.__version__)"
```

What it does: temp files to scratch; PyTorch with CUDA 12.6; the project requirements
(without SAM2, which `cutie_composite` does not use); `transformers` 5 (SAM3 lives there);
CUTIE cloned into `models/cutie` and installed (falls back to inference-only dependencies
if the full install fails); CUTIE weights into `models/cutie/weights/`; an import check
that prints `ALL OK`.

Verified 2026-10-21: ends with `ALL OK 2.14.1+cu126 5.20.2` (torch, transformers).
The full CUTIE install worked (no fallback needed). CUTIE weights downloaded to
`models/cutie/weights/` (`coco_lvis_h18_itermask.pth`, `cutie-base-mega.pth`).
A `torch.jit.script is deprecated` FutureWarning is harmless.

## 7. Hugging Face token and SAM3 weights

`cutie_composite` uses SAM3 on the first 125 frames, so this step is required.
`facebook/sam3` is gated: Meta has to approve each Hugging Face account.

1. Open https://huggingface.co/facebook/sam3, log in, and submit the access form.
   The page then says "awaiting a review from the repository authors" (submitted 2026-10-21,
   pending). Status: https://huggingface.co/settings/gated-repos
2. Create a token at https://huggingface.co/settings/tokens ("Create new token", type Read).
   It starts with `hf_` and is shown only once. The model page itself never shows a token.
3. Store it on Bunya:

```bash
echo hf_YOUR_REAL_TOKEN > ~/.hf_token && chmod 600 ~/.hf_token
```

4. After Meta approves, download SAM3 (about 3 GB, to `~/.cache/huggingface`):

```bash
rat
export TMPDIR=/scratch/user/uqcvaras/tmp
python -c "import os; from huggingface_hub import snapshot_download; print(snapshot_download('facebook/sam3', token=open(os.path.expanduser('~/.hf_token')).read().strip()))"
```

Meta approved access on 2026-10-21 (same day, a few hours after the request).

(download not yet run)

## 8. Run cutie_composite + evidence (needs the Slurm account)

```bash
rat && git pull
sbatch slurm/run_cutie.sbatch data/raw/original.mp4 2      # video, number of animals
squeue -u $USER
tail -f outputs/slurm/cutie_<JOBID>.out
```

The job writes `outputs/runs/<timestamp>_cutie_composite/` with `overlays/` (tracking
videos), `contacts/` (`results.xlsx`, `report.pdf`, `event_log.txt`), `validation/`
(evidence clips, thumbnails, `review.xlsx`, `index.html`) and a zip of the whole run,
then prints the scp line to download it.

Checked off-cluster on 2026-10-21: the contact chain after tracking (contacts_v2,
post-processing, Excel/PDF report, validation sample, scoring, zip) ran end to end on
synthetic sessions from `scripts/make_demo_session.py`: 11 evidence clips for 120 s,
2 for 10 s. SAM3 + CUTIE themselves can only run on a GPU node.

(pending Slurm account)

## Every new login

The module and the hook do not persist between sessions:

```bash
cd ~/computer-vision-tracking
module load miniforge/26.3.2-1
eval "$(conda shell.bash hook)"
conda activate rattrack
```

Shortcut, added once to `~/.bashrc` so every later login only needs `rat`:

```bash
echo 'alias rat="cd ~/computer-vision-tracking && module load miniforge/26.3.2-1 && eval \"\$(conda shell.bash hook)\" && conda activate rattrack"' >> ~/.bashrc
source ~/.bashrc
```

## Pending: Slurm account

Without a Slurm account, `sbatch`, `salloc` and `srun` fail with
`Invalid account or account/partition combination specified`.
`sacctmgr -n show assoc user=uqcvaras format=account%30` returns nothing: the staff account
has no association. A group admin (listed in the RCC approval email) has to add `uqcvaras`
to the group's Slurm account. The previous student account (`s4948012`) had a default one,
which is why jobs used to run without `--account`.
