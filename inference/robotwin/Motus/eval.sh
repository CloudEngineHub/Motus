#!/bin/bash

policy_name=Motus
task_name=beat_block_hammer
task_config=demo_clean
ckpt_setting=${3:-""} # Note: The folder path where Motus checkpoint(mp_rank_00_model_states.pt) is located.
seed=42
gpu_id=5
# LAWM checkpoint path
export LAWM_CHECKPOINT=${ckpt_setting}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} 
    # [TODO] add parameters here
