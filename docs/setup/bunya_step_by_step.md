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

Verified 2026-10-21: 12 files, 6.55 GB, about 1 minute, cached at
`~/.cache/huggingface/hub/models--facebook--sam3/`. Never paste the token into chats,
issues or repo files; if it leaks, delete it at https://huggingface.co/settings/tokens,
create a new one and rewrite `~/.hf_token` (the cached model stays valid).

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

First run, 2026-10-22 22:09 (job 30407392, 5 s test): Slurm accepted the job, A100 80GB
(MIG 4g.40gb slice), `torch ... cuda True`. It then failed at import with
`ModuleNotFoundError: No module named 'sam2'`: `src/common/model_loaders.py` imported SAM2
at module level although `cutie_composite` never uses it. Fixed by importing SAM2 lazily;
the full `cutie_composite` import chain was then checked without SAM2 installed.

Waiting for the Slurm sync after joining the group (about 1 h) without polling by hand:

```bash
nohup bash -c 'until sacctmgr -n show assoc user=uqcvaras | grep -q a_qbi; do sleep 300; done; date; sbatch slurm/run_cutie.sbatch data/raw/original.mp4 2 5' > ~/wait_submit.log 2>&1 &
cat ~/wait_submit.log      # 'Submitted batch job N' once it went through
```

SAM3 loads offline on the compute node: the job sets `HF_HUB_OFFLINE=1` (the model is in
`~/.cache/huggingface` from step 7) and a failed Hugging Face login only logs a warning.
The job's `.out`/`.err` files appear only once the job leaves `PD` (pending) and starts.

Second run, 2026-10-23 (job 30428224, 5 s = 125 frames): the whole pipeline ran on the GPU.
SAM3 + CUTIE tracked both mice, pose detected in 125/125 frames for each, 0 % carry-over,
report.pdf and the zip were written (2.2 MB). Two issues:

- `Contact post-processing failed: No module named 'scripts.postprocess_contacts_simple'`.
  The editable CUTIE install puts `models/cutie` on sys.path, and Cutie ships its own
  `scripts/__init__.py`, which shadowed this repo's `scripts/` (a namespace package).
  Fixed by adding `scripts/__init__.py` and `src/__init__.py`; reproduced and verified locally.
- 0 contact bouts in 5 s, so the validation sample had nothing to draw from (expected for
  such a short clip; the job continues and still zips the run).

Third run, 2026-10-23 (job 30428396, full 10 s = 250 frames): everything worked end to end.
`Handoff: 2/2`, 0 % carry-over, post-processing ran (the `scripts` fix works), results.xlsx,
report.pdf, event_log.txt and a 4.2 MB zip. Contacts: 4 raw bouts, 0.6 s in total, all
shorter than the 0.3 s minimum, so 0 real contact events and the whole clip is one NC event.
The validation sample therefore reports "No events to sample": not a bug, the clip has no
contact. Evidence needs a longer stretch of the original video. To inspect the raw bouts:

```bash
python scripts/validate_contacts.py sample outputs/runs/<run> --source bouts
```

A harmless `transformers` notice says the optional `kernels` package is missing (mask
post-processing extras are skipped); tracking still succeeded.

Fourth run, 2026-10-23 (job 30441656, `multimicevideo.avi`, 6 mice, first 10 s = 300
frames at 30 fps): `Handoff: 6/6`, pose in 300/300 frames for every animal, 15 pairs, 86
raw bouts -> 46 cleaned events (16 contacts: 7 N2AG, 5 SBS, 3 N2N, 1 FOL), 16 evidence
clips, zip downloaded and opened on the laptop. First multi-animal run end to end.

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

Auto-run it on every interactive login (once). The `$- == *i*` guard keeps it out of
scp and Slurm job shells:

```bash
echo '[[ $- == *i* ]] && rat' >> ~/.bashrc
```

## Pending: Slurm account

Without a Slurm account, `sbatch`, `salloc` and `srun` fail with
`Invalid account or account/partition combination specified`.
`sacctmgr -n show assoc user=uqcvaras format=account%30` returns nothing: the staff account
has no association. A group admin (listed in the RCC approval email) has to add `uqcvaras`
to the group's Slurm account. The previous student account (`s4948012`) had a default one,
which is why jobs used to run without `--account`.

Attempts so far:

| Date | Command | Result |
|------|---------|--------|
| 2026-10-21 | `sacctmgr -n show assoc user=uqcvaras format=account%30` | empty (no association) |
| 2026-10-21 | `sbatch --test-only slurm/run_cutie.sbatch ...` | `Invalid account or account/partition combination specified` |
| 2026-10-21 | `sbatch slurm/run_cutie.sbatch data/raw/original.mp4 2` | same error |
| 2026-10-21 | old interactive way: `salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00` | same error |
| 2026-10-21 | minimal request, no GPU, default partition: `salloc --time=00:01:00` | same error |

Conclusion: no Slurm request of any kind is accepted. Only a group admin can fix it.

How RCC says it works (UQ-RCC hpc-docs: `guides/Accounting-group-admin.md`,
`policy/Bunya-Accounting-Group-Operational-Procedure.md`, read 2026-10-21):

- Every Bunya user must be in exactly one accounting group; every job needs its
  `--account=a_...` string. There is no default or school-level group.
- Users are added by the group's owner or administrator (usually the supervisor, group
  leader or group admin). Owners/admins see the AccountString in the QRIScloud portal
  (https://www.qriscloud.org.au/ -> Account -> Services Dashboard -> Groups).
- RCC only adds a member after written request or confirmation from the group owner.
- After being added: log in again and run `groups`; Slurm syncs new members hourly, so
  allow about 1 hour before the first job.
- RCC Service Desk: rcc-support@uq.edu.au.

2026-10-22: added to the QBI accounting group ("QBI Group", Standard role) after an
email to a group admin from the Balbi team. Next: log in again, wait about 1 hour for the
Slurm sync, then `groups` shows the `a_...` string.
Confirmed the same day: `groups` -> `qris-uq a_qbi`. The AccountString is **`a_qbi`**; every
`.sbatch` file carries `#SBATCH --account=a_qbi`. For salloc add `--account=a_qbi`.

Who to ask, in order: the supervisor (which accounting group does the lab use, and who
owns it), then the group owner/admin to add `uqcvaras`; if nobody knows, RCC support.

If the interactive way gets a node (the prompt changes from `bunyaN` to a compute node),
the pipeline runs by hand:

```bash
rat
nvidia-smi -L
export HF_TOKEN=$(cat ~/.hf_token)
python -m src.pipelines.cutie_composite.run --config configs/hpc_cutie_composite.yaml video_path=data/raw/original.mp4 detection.max_animals=2
```
