#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=6
MODEL_NAME="QWen3VL"
MODEL_PATH="/data/fengbailong-20250924/a4sben/models/Qwen3-VL-8B-Instruct"
VIDEO_DIR="/data/fengbailong-20250924/AI4Service_Benchmark/data/EgoLife/A1_JAKE/First_1h"
RESULT_DIR="../../results"

echo "QWen3VL Inference"
echo "model_path: $MODEL_PATH"
echo "video_dir: $VIDEO_DIR"
echo "result_dir: $RESULT_DIR/$MODEL_NAME"

# run Python script
python3 ../../inference.py \
    --model $MODEL_NAME \
    --video_dir $VIDEO_DIR \
    --result_dir $RESULT_DIR \
    --model_path $MODEL_PATH \
    $OPTS

echo "Inference completed."
echo "result_dir: $RESULT_DIR/$MODEL_NAME/"