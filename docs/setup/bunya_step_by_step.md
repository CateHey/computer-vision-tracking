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
```

(in progress)

## Pending: Slurm account

Without a Slurm account, `sbatch`, `salloc` and `srun` fail with
`Invalid account or account/partition combination specified`.
`sacctmgr -n show assoc user=uqcvaras format=account%30` returns nothing: the staff account
has no association. A group admin (listed in the RCC approval email) has to add `uqcvaras`
to the group's Slurm account. The previous student account (`s4948012`) had a default one,
which is why jobs used to run without `--account`.
