#!/bin/bash
# Local training script for direct execution without SLURM
# Usage: bash scripts/run_local.sh [CONFIG_FILE] [RUN_NAME] [GPU_IDS] [RESUME_FROM] [PRETRAIN_CKPT]
# Example: bash scripts/run_local.sh configs/robotwin.yaml test_run "0,1,2,3"
# Finetune Example: bash scripts/run_local.sh configs/ac_one.yaml ac_one_finetune "0,1,2,3,4,5,6,7" "" /path/to/pretrain/checkpoint

echo "Starting local training on $(hostname) at $(date)"

# Setup environment
# PROJECT_ROOT="/share/home/bhz/test/latent_action_world_model/lawm"
# cd $PROJECT_ROOT

# Load modules and activate conda environment
module load cuda/12.8 2>/dev/null || echo "Warning: Could not load CUDA module (may not be needed)"
source /share/home/bhz/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-predict2-hb

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "cosmos-predict2-hb" ]]; then
    echo "Error: Failed to activate cosmos-predict2-hb environment"
    exit 1
fi

# Set environment variables
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export OMP_NUM_THREADS=8
export CUDA_HOME=$CONDA_PREFIX

# Parse command line arguments
# CONFIG_FILE=${1:-${CONFIG_FILE:-"configs/latent_action.yaml"}}
# RUN_NAME=${2:-${RUN_NAME:-"latent_action_pretrain_test"}}
GPU_IDS=${3:-${GPU_IDS:-"0,1,2,3,4,5,6,7"}}  # Default to GPU 0
CONFIG_FILE=${CONFIG_FILE:-"configs/robotwin.yaml"}
RUN_NAME=${RUN_NAME:-"robotwin"}
# PRETRAIN_CKPT: leave empty by default; prefer YAML training.pretrain_ckpt or explicit CLI
PRETRAIN_CKPT=${5:-${PRETRAIN_CKPT:-""}}
RESUME_FROM=${RESUME_FROM:-"/share/home/bhz/motus-robotics/Motus/checkpoints/robotwin/robotwin/checkpoint_step_50"}

# Set CUDA_VISIBLE_DEVICES to specify which GPUs to use
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Count the number of GPUs specified
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=========================================="
echo "GPU Configuration"
echo "=========================================="
echo "Specified GPU IDs: $GPU_IDS"
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Validate GPU IDs
echo "Validating GPU availability..."
for gpu_id in "${GPU_ARRAY[@]}"; do
    if ! nvidia-smi -i $gpu_id >/dev/null 2>&1; then
        echo "Error: GPU $gpu_id is not available or invalid"
        echo "Available GPUs:"
        nvidia-smi -L
        exit 1
    else
        echo "✅ GPU $gpu_id is available"
    fi
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -la configs/
    exit 1
fi

# NCCL settings for better performance (only for multi-GPU)
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=bond1
    export NCCL_IB_RETRY_CNT=7
    export NCCL_IB_TIMEOUT=23
    export NCCL_DEBUG=WARN  # Less verbose than INFO for local runs
    
    # Increase timeout for checkpoint saving
    export NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
fi

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Local Training Configuration"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Config File: $CONFIG_FILE"
echo "Run Name: $RUN_NAME"
echo "GPU IDs: $GPU_IDS"
echo "Number of GPUs: $NUM_GPUS"
echo "Resume From (YAML): will use resume.checkpoint_path"
echo "Finetune From (YAML): will use finetune.checkpoint_path"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python Path: $(which python)"
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'CUDA not found in PATH')"
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -n $NUM_GPUS
echo "=========================================="

# Check if training script exists
if [ ! -f "train/train.py" ]; then
    echo "Error: Training script not found: train/train.py"
    exit 1
fi

# Log file for this run
LOG_FILE="logs/local_${RUN_NAME}_gpus_${GPU_IDS//,/_}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training... (logs will be saved to $LOG_FILE)"
echo "Use 'tail -f $LOG_FILE' to monitor progress"
echo "Use 'Ctrl+C' to stop training"

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "Training interrupted at $(date)"
    echo "Logs saved to: $LOG_FILE"
    exit 0
}

# Set trap for graceful shutdown
trap cleanup SIGINT SIGTERM

# Choose training command based on number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single-GPU training on GPU $GPU_IDS..."
    # Single GPU training
    TRAIN_CMD="python train/train.py --config $CONFIG_FILE --run_name $RUN_NAME --report_to tensorboard"
    eval $TRAIN_CMD 2>&1 | tee $LOG_FILE
else
    echo "Running multi-GPU training on GPUs $GPU_IDS with DeepSpeed..."
    # Multi-GPU training with DeepSpeed
    TRAIN_CMD="deepspeed --include localhost:$GPU_IDS --master_port 29500 train/train.py --deepspeed configs/zero1.json --config $CONFIG_FILE --run_name $RUN_NAME --report_to tensorboard"
    eval $TRAIN_CMD 2>&1 | tee $LOG_FILE
fi

# Check training exit status
TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully at $(date)"
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE at $(date)"
fi
echo "Logs saved to: $LOG_FILE"
echo "=========================================="