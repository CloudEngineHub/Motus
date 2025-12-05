#!/bin/bash
# SLURM script for multi-node distributed training
# Usage: sbatch scripts/slurm_multi_node.sh
# Resume from checkpoint: RESUME_FROM="checkpoints/robotwin/robotwin_uni_0930_pad_speed/checkpoint_step_40000" sbatch scripts/slurm_multi_node.sh
# Finetune with pretrain: PRETRAIN_CKPT="checkpoints/latent_action/latent_action_pretrain_test/checkpoint_step_20000" sbatch scripts/slurm_multi_node.sh

#SBATCH --job-name=motus_multi
#SBATCH --output=/share/home/bhz/motus/logs_stage4/slurm_multi_%j.out
#SBATCH --error=/share/home/bhz/motus/logs_stage4/slurm_multi_%j.err
#SBATCH --nodes=2

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=256
#SBATCH --mem=1500G
#SBATCH --partition=emb
#SBATCH --exclusive

echo "Starting multi-node job on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_NODEID: $SLURM_NODEID"

# Setup environment
PROJECT_ROOT="/share/home/bhz/motus-robotics/Motus"
cd $PROJECT_ROOT

# Load modules and activate conda environment
module load cuda/12.8 || echo "Warning: Could not load CUDA module"
source /share/home/bhz/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-predict2-hb

# Set environment variables
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export OMP_NUM_THREADS=8
export CUDA_HOME=$CONDA_PREFIX

# Get node information
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
master_addr=$(echo "$nodes" | head -n 1)

echo "NODELIST: $nodes"
echo "MASTER_ADDR: $master_addr"
echo "Current node index: $SLURM_NODEID"

# NCCL settings for multi-node
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_13:1,mlx5_16:1,mlx5_17:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=23
export NCCL_DEBUG=INFO

# Increase timeout for checkpoint saving (default is 600s/10min, set to 30min)
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# Create logs directory
mkdir -p /share/home/bhz/motus/logs_stage4

# Configuration
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_FILE=${CONFIG_FILE:-"configs/robotwin.yaml"}
RUN_NAME=${RUN_NAME:-"robotwin_multi"}

echo "=========================================="
echo "Multi-Node Training Configuration"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"
echo "Master addr: $master_addr"
echo "Master port: $MASTER_PORT"
echo "Config: $CONFIG_FILE"
echo "Run name: $RUN_NAME"
echo "Resume From (YAML): resume.checkpoint_path"
echo "Finetune From (YAML): finetune.checkpoint_path"
echo "=========================================="

# Multi-node distributed training - launch torchrun on all nodes via srun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$master_addr \
    --master_port=$MASTER_PORT \
    train/train.py \
    --deepspeed configs/zero1.json \
    --config $CONFIG_FILE \
    $(if [ -n "$RUN_NAME" ]; then echo "--run_name $RUN_NAME"; fi) \
    --report_to tensorboard

echo "Training completed at $(date)"