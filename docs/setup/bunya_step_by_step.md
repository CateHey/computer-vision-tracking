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

## 6. Install PyTorch

(in progress)

## Every new login

The module and the hook do not persist between sessions:

```bash
cd ~/computer-vision-tracking
module load miniforge/26.3.2-1
eval "$(conda shell.bash hook)"
conda activate rattrack
```

## Pending: Slurm account

Without a Slurm account, `sbatch`, `salloc` and `srun` fail with
`Invalid account or account/partition combination specified`.
`sacctmgr -n show assoc user=uqcvaras format=account%30` returns nothing: the staff account
has no association. A group admin (listed in the RCC approval email) has to add `uqcvaras`
to the group's Slurm account. The previous student account (`s4948012`) had a default one,
which is why jobs used to run without `--account`.
