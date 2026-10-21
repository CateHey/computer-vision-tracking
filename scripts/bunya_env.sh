#!/bin/bash
# ============================================================================
# Bunya (UQ HPC) settings — the single place every HPC script reads from.
# ============================================================================
#
# On Bunya, after cloning:
#   cd ~/computer-vision-tracking && source scripts/bunya_env.sh
#
# It then provides:
#   bunya_activate          load the miniforge module + activate the conda env
#   bunya_gpu_shell N H     interactive shell on a GPU node (N GPUs, H hours)
#   bunya_require_account   fail fast when no Slurm account string is known
#   HF_TOKEN                read from ~/.hf_token when not already exported (SAM3)
#
# Every sbatch script in slurm/ and scripts/ sources this file itself. The only
# thing to do by hand is source it once per login, so that salloc and sbatch
# pick up the account string (exported as SBATCH_ACCOUNT / SALLOC_ACCOUNT).
#
# Override any value by exporting it before sourcing:
#   BUNYA_ACCOUNT=a_mygroup source scripts/bunya_env.sh
#
# Safe under `set -euo pipefail` (every read uses ${VAR:-}).
# ============================================================================

# --- Identity -----------------------------------------------------------------
export BUNYA_USER="${BUNYA_USER:-uqcvaras}"
export BUNYA_HOST="${BUNYA_HOST:-bunya.rcc.uq.edu.au}"

# --- Slurm account string (a_<group>) -----------------------------------------
# Bunya rejects every job that has no --account. The string is handed out by
# your group's Bunya admin (the admin list is in the RCC approval email). Once they add you it
# shows up in `groups`; if exactly one a_* group is listed it is used here.
export BUNYA_ACCOUNT="${BUNYA_ACCOUNT:-}"
if [ -z "$BUNYA_ACCOUNT" ] && command -v groups >/dev/null 2>&1; then
    _bunya_accts="$(groups 2>/dev/null | tr ' ' '\n' | grep -E '^a_' || true)"
    _bunya_n="$(printf '%s\n' "$_bunya_accts" | grep -c . || true)"
    if [ "$_bunya_n" -eq 1 ]; then
        export BUNYA_ACCOUNT="$_bunya_accts"
    elif [ "$_bunya_n" -gt 1 ]; then
        echo "bunya_env: several account strings in \`groups\`; pick one with:" >&2
        echo "           export BUNYA_ACCOUNT=<one of: $(echo $_bunya_accts)>" >&2
    fi
    unset _bunya_accts _bunya_n
fi
if [ -n "$BUNYA_ACCOUNT" ]; then
    # Slurm reads these as the default --account for sbatch / salloc / srun.
    export SBATCH_ACCOUNT="$BUNYA_ACCOUNT"
    export SALLOC_ACCOUNT="$BUNYA_ACCOUNT"
    export SLURM_ACCOUNT="$BUNYA_ACCOUNT"
elif command -v sbatch >/dev/null 2>&1; then
    echo "bunya_env: WARNING: no account string yet (BUNYA_ACCOUNT is empty)." >&2
    echo "           Jobs are rejected until a group admin adds you; then re-source this file." >&2
fi

# --- Paths --------------------------------------------------------------------
# /home         (50 GB, never purged)                 : code, conda env, weights
# /scratch/user (300 GB, purged after 90 days idle)  : videos + run outputs
# scripts/bunya_setup.sh symlinks data/raw and outputs/runs into the scratch dir,
# so every other script keeps using relative paths. outputs/slurm (job logs) stays in home.
export BUNYA_PROJECT_DIR="${BUNYA_PROJECT_DIR:-$HOME/computer-vision-tracking}"
export BUNYA_SCRATCH_DIR="${BUNYA_SCRATCH_DIR:-/scratch/user/${USER:-$BUNYA_USER}/computer-vision-tracking}"

# --- Python environment -------------------------------------------------------
# RCC's supported route is conda through the miniforge module. If the pinned
# version disappears, bunya_load_conda falls back to the module's default.
# Check with:  module avail miniforge
export BUNYA_CONDA_MODULE="${BUNYA_CONDA_MODULE:-miniforge/26.1.0-0}"
export BUNYA_CONDA_ENV="${BUNYA_CONDA_ENV:-rattrack}"

# --- Hugging Face token (SAM3 = facebook/sam3 is a gated model) ----------------
# Create a read token at https://huggingface.co/settings/tokens, accept the terms
# at https://huggingface.co/facebook/sam3 with the same account, then store it:
#   echo hf_xxx > ~/.hf_token && chmod 600 ~/.hf_token
# An exported HF_TOKEN always wins over the file. sbatch forwards the environment,
# and every sbatch script sources this file, so jobs see it either way.
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
fi

# --- Windows side: where the printed scp download commands point ---------------
export BUNYA_LOCAL_DIR="${BUNYA_LOCAL_DIR:-C:\Users\uqcvaras\Downloads\computer-vision-tracking}"

# --- Helpers ------------------------------------------------------------------

# Make the `module` command available in non-login shells.
_bunya_ensure_module() {
    if type module >/dev/null 2>&1; then return 0; fi
    local f
    for f in /etc/profile.d/z00_lmod.sh /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
             /usr/share/lmod/lmod/init/bash; do
        if [ -r "$f" ]; then
            # shellcheck disable=SC1090
            source "$f"
            return 0
        fi
    done
    echo "bunya_env: cannot find the Lmod init script; use a login shell (bash -l)" >&2
    return 1
}

# Load the miniforge module and make `conda` usable (no env activated yet).
bunya_load_conda() {
    local had_u=0 root
    case "$-" in *u*) had_u=1 ;; esac
    set +u   # Lmod and conda hooks are not always clean under set -u
    if ! _bunya_ensure_module; then
        if [ "$had_u" -eq 1 ]; then set -u; fi
        return 1
    fi
    if ! module load "$BUNYA_CONDA_MODULE" 2>/dev/null && ! module load miniforge; then
        echo "bunya_env: no miniforge module could be loaded (try: module avail miniforge)" >&2
        if [ "$had_u" -eq 1 ]; then set -u; fi
        return 1
    fi
    # RCC guide: source conda.sh instead of `conda init` (which breaks onBunya).
    root="${ROOTMINIFORGE:-${EBROOTMINIFORGE:-}}"
    if [ -z "$root" ] && command -v conda >/dev/null 2>&1; then
        root="$(dirname "$(dirname "$(command -v conda)")")"
    fi
    # shellcheck disable=SC1091
    source "$root/etc/profile.d/conda.sh"
    if [ "$had_u" -eq 1 ]; then set -u; fi
}

# Activate the project environment (conda env, or .venv when BUNYA_USE_VENV=1).
bunya_activate() {
    if [ "${BUNYA_USE_VENV:-0}" = "1" ]; then
        # shellcheck disable=SC1091
        source "$BUNYA_PROJECT_DIR/.venv/bin/activate"
        return
    fi
    bunya_load_conda || return 1
    local had_u=0
    case "$-" in *u*) had_u=1 ;; esac
    set +u
    if ! conda activate "$BUNYA_CONDA_ENV"; then
        echo "bunya_env: conda env '$BUNYA_CONDA_ENV' not found. Run: bash scripts/bunya_setup.sh" >&2
        if [ "$had_u" -eq 1 ]; then set -u; fi
        return 1
    fi
    if [ "$had_u" -eq 1 ]; then set -u; fi
}

# Fail fast when no Slurm account string is known.
bunya_require_account() {
    if [ -n "${BUNYA_ACCOUNT:-}" ]; then return 0; fi
    cat >&2 <<'EOF'
ERROR: no Slurm account string. Bunya rejects jobs without --account=a_<group>.
  1. Ask one of your group's Bunya admins to add you (the admin list is in the RCC approval email).
  2. Once added, `groups` lists the a_... string and bunya_env.sh picks it up.
  3. Or set it by hand:  export BUNYA_ACCOUNT=a_<group>
EOF
    return 1
}

# Interactive shell on a GPU node:  bunya_gpu_shell [gpus=1] [hours=6] [mem=64G]
# --qos=gpu allows at most 4 GPUs. Inside the new shell, source this file again
# (srun starts a clean login shell) and then run bunya_activate.
bunya_gpu_shell() {
    bunya_require_account || return 1
    local gpus="${1:-1}" hours="${2:-6}" mem="${3:-64G}"
    local cpus=$(( gpus * 4 ))
    echo "salloc: ${gpus} GPU(s), ${cpus} CPUs, ${mem}, ${hours}h, account ${BUNYA_ACCOUNT}" >&2
    salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task="$cpus" --mem="$mem" \
        --partition=gpu_cuda --qos=gpu --gres="gpu:${gpus}" \
        --time="${hours}:00:00" --job-name=cvt-shell --account="$BUNYA_ACCOUNT" \
        srun --export=PATH,TERM,HOME,LANG --pty /bin/bash -l
}

# Slurm needs the log directory to exist before sbatch, or the job dies at start.
if [ -d "$BUNYA_PROJECT_DIR" ]; then
    mkdir -p "$BUNYA_PROJECT_DIR/outputs/slurm" 2>/dev/null || true
fi
