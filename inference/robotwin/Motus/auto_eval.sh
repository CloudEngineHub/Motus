#!/bin/bash
# Auto evaluation script for Motus policy without SLURM

echo "Starting RoboTwin batch evaluation on $(hostname) at $(date)"

# Setup environment
PROJECT_ROOT="/share/home/bhz/test/RoboTwin"
# Robotwin folder path.
# PROJECT_ROOT="" Such as ".../RoboTwin"
cd $PROJECT_ROOT

# Note: Load modules and activate conda environment, modify as needed
module load cuda/12.8 || echo "Warning: Could not load CUDA module"
source /share/home/bhz/miniconda3/etc/profile.d/conda.sh
conda activate  /share/home/bhz/miniconda3/envs/RoboTwin-hb

# Set environment variables
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export OMP_NUM_THREADS=8

# Create logs directory
LOG_DIR="logs_robotwin"
export LOG_DIR
mkdir -p $LOG_DIR

# Configuration parameters
policy_name="Motus"
task_config="demo_randomized"
seed="42"
# Note: The folder path where Motus checkpoint(mp_rank_00_model_states.pt) is located. Such as .../checkpoint_step_40000
base_ckpt_path=""

# Load tasks from file generated from envs/*.py
tasks_file="policy/lawm/tasks_all.txt"
if [ ! -f "$tasks_file" ]; then
    echo "Tasks file not found: $tasks_file"
    exit 1
fi
mapfile -t tasks < "$tasks_file"


gpus=(0 1 2 3 4 5 6 7)

echo -e "\033[33m=== Batch evaluation start ===\033[0m"
echo "Num tasks: ${#tasks[@]}"
echo "Log directory: ${LOG_DIR}"
echo "Node: $(hostname)"

# Track per-GPU running process
declare -A gpu_pid

# Check if a PID is running
is_running() {
    if [ -z "$1" ]; then
        return 1
    fi
    kill -0 "$1" 2>/dev/null
}

# Find a free GPU; wait if all are busy
get_free_gpu() {
    while true; do
        for gpu_id in "${gpus[@]}"; do
            pid="${gpu_pid[$gpu_id]}"
            if ! is_running "$pid"; then
                echo "$gpu_id"
                return 0
            fi
        done
        sleep 2
    done
}

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local bar_length=50
    local filled_length=$((percent * bar_length / 100))
    
    printf "\r["
    for ((i=0; i<filled_length; i++)); do printf "="; done
    for ((i=filled_length; i<bar_length; i++)); do printf " "; done
    printf "] %d%% (%d/%d)" $percent $current $total
}

# Launch tasks, at most one per GPU concurrently
pids=()
completed_tasks=0
total_tasks=${#tasks[@]}

for task in "${tasks[@]}"; do
    gpu_id=$(get_free_gpu)
    ckpt_setting="${base_ckpt_path}/pytorch_model"

    echo -e "\n\033[32mLaunch task: ${task} | GPU: ${gpu_id} | Node: $(hostname)\033[0m"

    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        
        PYTHONWARNINGS=ignore::UserWarning \
        python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
            --overrides \
            --task_name "${task}" \
            --task_config "${task_config}" \
            --ckpt_setting "${ckpt_setting}" \
            --seed "${seed}" \
            --policy_name "${policy_name}" \
            --log_dir "${LOG_DIR}" \
            > ${LOG_DIR}/${task}.log 2>&1
        
        echo -e "\033[35mDone: ${task} | GPU: ${gpu_id} | Node: $(hostname)\033[0m"
    ) &
    pid=$!
    gpu_pid[$gpu_id]=$pid
    pids+=($pid)

    # Small delay to avoid burst launching
    sleep 1
done

echo -e "\n\033[33mAll tasks launched, waiting for completion...\033[0m"

# Wait for all background jobs with progress tracking
for pid in "${pids[@]}"; do
    wait "$pid"
    ((completed_tasks++))
    show_progress $completed_tasks $total_tasks
done

echo -e "\n\033[32m=== Batch evaluation completed at $(date) ===\033[0m"
echo "All ${total_tasks} tasks completed successfully"
echo "Logs saved in: ${LOG_DIR}/"

# Generate summary report
echo -e "\n\033[36m=== Generating summary report ===\033[0m"
summary_file="${LOG_DIR}/evaluation_summary_$(date +%Y%m%d_%H%M%S).txt"

cat > $summary_file << EOF
RoboTwin LAWM Evaluation Summary
================================
Date: $(date)
Host: $(hostname)
Checkpoint: ${base_ckpt_path}
Policy: ${policy_name}
Task Config: ${task_config}
Seed: ${seed}
Total Tasks: ${total_tasks}
GPUs Used: ${gpus[@]}

Task Results:
EOF

# Check each log file for success/failure
for task in "${tasks[@]}"; do
    log_pattern="${LOG_DIR}/batch_${task}_gpu*_*.log"
    log_file=$(ls $log_pattern 2>/dev/null | head -1)
    if [ -f "$log_file" ]; then
        if grep -q "Episode.*completed" "$log_file" 2>/dev/null; then
            echo "  ✅ $task: SUCCESS" >> $summary_file
        else
            echo "  ❌ $task: FAILED" >> $summary_file
        fi
    else
        echo "  ⚠️  $task: LOG NOT FOUND" >> $summary_file
    fi
done

echo "Summary report saved to: $summary_file"

# Show quick stats
success_count=$(grep "SUCCESS" $summary_file | wc -l)
failed_count=$(grep "FAILED" $summary_file | wc -l)
echo -e "\033[32mSuccessful tasks: $success_count\033[0m"
echo -e "\033[31mFailed tasks: $failed_count\033[0m"

if [ $failed_count -eq 0 ]; then
    echo -e "\033[32m🎉 All tasks completed successfully!\033[0m"
else
    echo -e "\033[33m⚠️  Some tasks failed. Check logs for details.\033[0m"
fi