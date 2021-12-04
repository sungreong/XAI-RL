#!/bin/bash

#source /home/conda/etc/profile.d/conda.sh
#conda activate xai
#cd /home/jovyan/nas/RL_TF_Team/felix/XAI/XAI-RL

echo "training..."
python train.py --algo PPO    --device cpu --seed 228725 --env_name BreakoutDeterministic-v4 &
sleep 2
python train.py --algo DQN    --device cpu --seed 228725 --env_name BreakoutDeterministic-v4 &
sleep 2
python train.py --algo Double --device cpu --seed 228725 --env_name BreakoutDeterministic-v4 &
sleep 2
python train.py --algo PER    --device cpu --seed 228725 --env_name BreakoutDeterministic-v4 &
