#!/bin/bash
# One sweep worker pinned to a GPU. Configs are claimed with mkdir, which is atomic on POSIX,
# so several workers can share the queue without running the same config twice. Ordered
# longest-processing-time-first so the workers finish at roughly the same time.
GPU=$1
D=/home/user/Quantization/precoding_quantization/non_lin_precoding
cd "$D" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU
CLAIMS="$D/exp_results/sweep_claims"
mkdir -p "$CLAIMS"
START=$(date +%s)
for cfg in "6 1" "6 2" "6 3" "4 1" "4 2" "4 3" "2 1" "2 2" "2 3"; do
  set -- $cfg; K=$1; B=$2
  if mkdir "$CLAIMS/K${K}_b${B}" 2>/dev/null; then
    echo ""
    echo "######## [GPU$GPU] K=$K bits=$B start $(date '+%F %T')  ($(( ($(date +%s)-START)/60 )) min in) ########"
    python train_sweep.py --K $K --bits $B --epochs 15 --n-train 100000 --logit-l2 0.1 \
      || echo "!!! [GPU$GPU] K=$K b=$B FAILED, continuing"
  else
    echo "[GPU$GPU] K=$K b=$B taken by the other worker, skipping"
  fi
done
echo "=== [GPU$GPU] worker finished after $(( ($(date +%s)-START)/60 )) min ==="
