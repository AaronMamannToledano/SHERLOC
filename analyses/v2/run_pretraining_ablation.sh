#!/bin/bash
# two sequential queues (one per core); restartable (finished chunks are skipped)
cd "$(dirname "$0")"
( for v in no_encoder scratch_trained pretrained_unfrozen; do PYTHONUNBUFFERED=1 python pretraining_ablation.py impower150_4tp $v; done ) > results/pretraining_ablation/log_impower.txt 2>&1 &
( for v in no_encoder scratch_trained pretrained_unfrozen; do PYTHONUNBUFFERED=1 python pretraining_ablation.py pada1 $v; done ) > results/pretraining_ablation/log_pada1.txt 2>&1 &
wait
