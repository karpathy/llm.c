#!/bin/bash
#SBATCH --job-name=llmc-multinode
#SBATCH --output=/dfs/llm.c/log124M/%x_%j_%t.log
#SBATCH --ntasks=32                                 # total number of processes to launch 
#SBATCH --ntasks-per-node=8                         # assuming each node has 8 gpus
#SBATCH --gres=gpu:8                                # request 8 gpus from each node
#SBATCH --nodelist=node[000-003]                    # list of the nodes to dispatch processes (32/8=4)

cd /dfs/llm.c/                                      # path to the repo in distributed file system
mkdir -p log124M

# export NCCL_SOCKET_IFNAME=ib0                     # network interface Ethernet or InifiniBand which enables gpu direct rdma
# export NCCL_IB_HCA=mlx5_0,mlx5_1                  # list of all InfiniBand devices available if available

# GPT-2 (124M) repro on FineWeb100B
# Global batch size is set to (1024 * 64) * 32
srun bash -c "
    ./train_gpt2cu \
    -i 'dev/data/fineweb100B/fineweb_train_*.bin' \
    -j 'dev/data/fineweb100B/fineweb_val_*.bin' \
    -o "log124M" \
    -v 250 -s 20000 -g 144 \
    -h 1 \
    -b 64 -t 1024 \
    -d 2097152 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 10000 \
    -y 1 \
    -e d12 \
    -pn 32 \
    -pr \$SLURM_PROCID \
    -pg 8 \
    -pd "/dfs/llm.c/log124M""
