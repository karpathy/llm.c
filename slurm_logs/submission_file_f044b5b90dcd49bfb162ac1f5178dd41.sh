#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/work/dlclarge1/swelamo-LLMs/llm.c_project/llm.c/slurm_logs/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=submitit
#SBATCH --mem=6GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/work/dlclarge1/swelamo-LLMs/llm.c_project/llm.c/slurm_logs/%j_0_log.out
#SBATCH --partition=mlhiwidlc_gpu-rtx2080
#SBATCH --signal=USR2@180
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /work/dlclarge1/swelamo-LLMs/llm.c_project/llm.c/slurm_logs/%j_%t_log.out --error /work/dlclarge1/swelamo-LLMs/llm.c_project/llm.c/slurm_logs/%j_%t_log.err /home/swelamo/miniconda3/envs/LLMs/bin/python -u -m submitit.core._submit /work/dlclarge1/swelamo-LLMs/llm.c_project/llm.c/slurm_logs
