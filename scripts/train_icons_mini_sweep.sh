#!/usr/bin/env bash
set -euo pipefail

export TORCH_CUDA_ARCH_LIST="7.5"
export CUDA_VISIBLE_DEVICES=0

BASE_FOLDER='/mnt/storage/Documents/project_envs/stylegan2-ada-pytorch'
OUT_ROOT='/mnt/data1/pretrained_models/stylegan2-training/icons/mini_sweep/'
mkdir -p "$OUT_ROOT"

# Dataset / settings
dataset_path='/mnt/data1/images/icons/img_collection/original_inverted'
mirror_x=False
mirror_y=False
conditional=1

config='24gb-gpu-complex'
snapshot_count=50
metrics=fid50k_full
num_workers=4
batch_size=16

# Hyperparameter grid
gamma_list=(1.0 3.28 6.0)
g_lr_list=(0.0018 0.0020 0.0022)
d_lr_list=(0.0022 0.0024 0.0028)

# Pilot / full budgets
kimg_pilot=1000    # run pilot to 1,000 kimg
kimg_full=5000     # resume to this total

# How many top configs to continue
top_k=2

# Phase 1: pilot runs
echo "=== Phase 1: Pilot stage to ${kimg_pilot} kimg ==="
for gamma in "${gamma_list[@]}"; do
  for gl in "${g_lr_list[@]}"; do
    for dl in "${d_lr_list[@]}"; do
      runname="pilot_g${gamma}_G${gl}_D${dl}"
      runout="${OUT_ROOT}/${runname}"
      mkdir -p "${runout}"

      cmd="python ${BASE_FOLDER}/train.py \
        --gpus=1 \
        --cfg=${config} \
        --metrics=${metrics} \
        --outdir=${runout} \
        --data=${dataset_path} \
        --cond=${conditional} \
        --snap=${snapshot_count} \
        --aug=noaug \
        --gamma=${gamma} \
        --batch=${batch_size} \
        --glrate=${gl} \
        --dlrate=${dl} \
        --mirror=${mirror_x} \
        --mirrory=${mirror_y} \
        --kimg=${kimg_pilot} \
        --workers=${num_workers}"
      
      echo "Launching pilot run: $runname"
      $cmd > "${runout}/train.log" 2>&1
    done
  done
done

echo "Pilot stage complete. Parsing results..."

# Phase 2: select top configs
declare -a winners=()
declare -a winner_dirs=()

# Loop over pilot run folders to read FID
for gamma in "${gamma_list[@]}"; do
  for gl in "${g_lr_list[@]}"; do
    for dl in "${d_lr_list[@]}"; do
      runname="pilot_g${gamma}_G${gl}_D${dl}"
      runout="${OUT_ROOT}/${runname}"
      fidfile="${runout}/metric-fid50k_full.jsonl"
      if [[ -f "${fidfile}" ]]; then
        # Read last line’s FID value
        # We assume JSON lines like {"kimg":..., "fid50k_full": ...}
        lastfid=$(jq -r 'select(.fid50k_full) | .fid50k_full' "${fidfile}" | tail -n1)
        echo "Run ${runname}: FID = ${lastfid}"
        winners+=( "$lastfid" )
        winner_dirs+=( "${runout}" )
      fi
    done
  done
done

# Sort winners by FID ascending (lower is better)
# We'll collect indices of top_k lowest FID
# We do a simple bubble logic for small arrays
n=${#winners[@]}
echo "Total completed pilots: ${n}"
for ((i=0; i<top_k; i++)); do
  best_idx=-1
  best_val=1e12
  for ((j=0; j<n; j++)); do
    val=${winners[j]}
    if (( $(echo "${val} < ${best_val}" | bc -l) )); then
      best_val=${val}
      best_idx=${j}
    fi
  done
  echo "Selected winner ${i}: dir=${winner_dirs[best_idx]} with FID=${best_val}"
  # Save that as one of the winners
  winners[$best_idx]=1e12   # exclude so we pick next
  # Store selected runout dir
  winner_subdir="${winner_dirs[best_idx]}"
  winners_directories+=("${winner_subdir}")
done

# Phase 3: resume / extend the winners to full budget
echo "=== Phase 3: resuming top ${top_k} runs to ${kimg_full} kimg ==="
for runout in "${winners_directories[@]}"; do
  # We need to find the best snapshot to resume from
  # The train.py writes snapshots like network-snapshot-xxxxx.pkl under runout
  # Find the highest kimg snapshot
  best_pkl=$(ls ${runout}/network-snapshot-*.pkl 2>/dev/null | sort | tail -n1)
  if [[ -z "${best_pkl}" ]]; then
    echo "No snapshot found in ${runout}, skipping resume"
    continue
  fi

  echo "Resuming ${runout} from ${best_pkl}"
  # Extract hyperparams from folder name or store them in a metadata file
  # For simplicity assume folder name encodes them: pilot_g{gamma}_G{gl}_D{dl}
  # Parse gamma, gl, dl from runout path
  base=$(basename "${runout}")
  # e.g. "pilot_g3.28_G0.0020_D0.0024"
  # Use shell parsing or regex
  IFS='_' read -ra parts <<< "${base}"
  gamma=${parts[1]#g}
  gl=${parts[2]#G}
  dl=${parts[3]#D}

  resumedir="${OUT_ROOT}/resume_${base}"
  mkdir -p "${resumedir}"

  cmd="python ${BASE_FOLDER}/train.py \
    --gpus=1 \
    --cfg=${config} \
    --metrics=${metrics} \
    --outdir=${resumedir} \
    --data=${dataset_path} \
    --cond=${conditional} \
    --snap=${snapshot_count} \
    --aug=noaug \
    --gamma=${gamma} \
    --batch=${batch_size} \
    --glrate=${gl} \
    --dlrate=${dl} \
    --mirror=${mirror_x} \
    --mirrory=${mirror_y} \
    --kimg=${kimg_full} \
    --nkimg=1000 \
    --workers=${num_workers} \
    --resume=${best_pkl}"

  # You may need to pass additional flag to indicate resume (some versions need `--resume_kimg`)
  # But many versions of train.py detect resume and pick up internal kimg automatically.

  echo "Launching resumed run: ${resumedir}"
  $cmd > "${resumedir}/train.log" 2>&1
done

echo "All resumed runs completed."
