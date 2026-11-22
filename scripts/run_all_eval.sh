#!/usr/bin/env bash
set -euo pipefail

SEED=42
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "OpenGVLab/InternVL2_5-2B"
    "llava-hf/llava-1.5-7b-hf"
)

PROMPTS=("p0" "p_soft" "p_strict")

mkdir -p results

for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Evaluating Model: ${MODEL} (Seed: ${SEED})"
    echo "=========================================="

    python src/evaluate.py \
        --model_name "${MODEL}" \
        --condition standard \
        --prompt_key p0 \
        --seed "${SEED}"

    for PROMPT in "${PROMPTS[@]}"; do
        python src/evaluate.py \
            --model_name "${MODEL}" \
            --condition conflict \
            --prompt_key "${PROMPT}" \
            --seed "${SEED}"
    done
done

echo "All evaluations completed successfully."
