#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate xai
cd /home/jovyan/nas/RL_TF_Team/felix/XAI/XAI-RL

echo "training..."
python train.py --algo PPO    --device cuda:0 --seed 228725 --env_name BreakoutDeterministic-v4 &
sleep 2
python train.py --algo DQN    --device cuda:1 --seed 228725 --env_name BreakoutDeterministic-v4 &
sleep 2
python train.py --algo Double --device cuda:2 --seed 228725 --env_name BreakoutDeterministic-v4 &
sleep 2
python train.py --algo PER    --device cuda:3 --seed 228725 --env_name BreakoutDeterministic-v4 &
