#!/bin/bash
# FSDP Training Launch Script
#
# Usage:
#   ./scripts/run_train_fsdp.sh --config configs/train/gsm8k_lora_fsdp.yaml
#   ./scripts/run_train_fsdp.sh --config configs/train/gsm8k_lora_fsdp.yaml --num_gpus 4
#
# This script wraps torchrun for convenient multi-GPU FSDP training.

set -e

# Default to all available GPUs
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --config <config.yaml> [--num_gpus N] [--master_port PORT]"
            echo ""
            echo "Arguments:"
            echo "  --config       Path to training config YAML (required)"
            echo "  --num_gpus     Number of GPUs to use (default: all available)"
            echo "  --master_port  Port for distributed communication (default: 29500)"
            echo ""
            echo "Example:"
            echo "  $0 --config configs/train/gsm8k_lora_fsdp.yaml --num_gpus 4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate config path
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: --config is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "=========================================="
echo "FSDP Training Launch"
echo "=========================================="
echo "Config:      $CONFIG_PATH"
echo "GPUs:        $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "=========================================="

# Launch with torchrun
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    -m scripts.run_train \
    --config "$CONFIG_PATH"
