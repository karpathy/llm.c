#!/bin/bash
#SBATCH --job-name=llmc-multinode                                     # job name
#SBATCH --output=/home/ubuntu/llm.c/scripts/multi_node/%x_%j_%t.log   # output file
#SBATCH --error=/home/ubuntu/llm.c/scripts/multi_node/%x_%j_%t.err    # error file
#SBATCH --partition=llmc                                              # Specify the GPU partition
#SBATCH --ntasks=16                                                   # total number of processes to launch on all nodes
#SBATCH --nodes=2                                                     # total number of nodes
#SBATCH --ntasks-per-node=8                                           # assuming each node has 8 gpus
#SBATCH --gres=gpu:8                                                  # request 8 gpus from each node

# NOTE: change the above slurm arguments to match your system!
# Run with `sbatch <path_to_this_script.sh>`

make train_gpt2cu USE_CUDNN=1 NO_USE_MPI=1

# NOTE: change the following to match your system
binary_path="/home/ubuntu/llm.c/train_gpt2cu"
out_dir="/ephemeral/data/fineweb/log_gpt2_124M_multi"
train_data_path='/ephemeral/data/fineweb/bin_10B/fineweb_train_*.bin'
val_data_path='/ephemeral/data/fineweb/bin_10B/fineweb_val_*.bin'
sync_fs_path=$out_dir  # needs to be a shared filesystem path that all nodes can access

# In case the file system is shared this is a no-op.
# Otherwise, we need to copy the binary to all nodes.
current_user=$USER
hosts=$(scontrol show hostnames $SLURM_JOB_NODELIST)  # get the hostnames of the allocated nodes
current_host=$(hostname)
for host in $hosts; do
    if [ $host == $current_host ]; then
        continue
    fi
    echo "copying $binary_path to $current_user@$host"
    scp -r $binary_path $current_user@$host:$binary_path
done

# Use this for NCCL debugging if you run into issues
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optimization flags
export NCCL_NET_GDR_LEVEL=2  # use GPUDirect RDMA - allows for direct memory access between GPUs across different nodes by bypassing the CPU
export NCCL_IB_DISABLE=0  # use InfiniBand if available

# NOTE: change the following environment variables to match your system - or comment them out if you don't need them
export NCCL_SOCKET_IFNAME=ens17
export OMPI_MCA_btl_tcp_if_include=ens17
export NCCL_P2P_LEVEL=PXB

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Make sure you're running in a SLURM environment. Did you forget to run with sbatch? Aborting."
    exit 1
else
    DATESTRING=`date "+%Y-%m-%dT%H:%M:%S"`
    echo "Running in a SLURM environment (job ID: $SLURM_JOB_ID, user: $current_user)"
    echo "Running on hosts: $(echo $(scontrol show hostname))"
    echo "$DATESTRING"
fi

srun -l -u bash -c "
    $binary_path \
    -i '$train_data_path' \
    -j '$val_data_path' \
    -o $out_dir \
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
    -n 5000 \
    -y 1 \
    -e d12 \
    -pn \$SLURM_NTASKS \
    -pr \$SLURM_PROCID \
    -pg \$SLURM_NTASKS_PER_NODE \
    -pf $sync_fs_path \
    -pi "fs" \
"

echo "$DATESTRING"