
make train_gpt2cu USE_CUDNN=1

# NOTE: change the following to match your system
binary_path="/home/ubuntu/llm.c/train_gpt2cu"
out_dir="/ephemeral/data/fineweb/log_gpt2_124M_multi"
train_data_path='/ephemeral/data/fineweb/bin_10B/fineweb_train_*.bin'
val_data_path='/ephemeral/data/fineweb/bin_10B/fineweb_val_*.bin'
# You can find these names either in `/etc/hosts`` file or in the terminal (user@host:~$).
host1="h100-node-1-0"  # master and worker node
host2="h100-node-1-1"  # worker node

# In case the file system is shared this is a no-op.
# Otherwise, we need to copy the binary to all nodes.
scp -r $binary_path $USER@$host2:$binary_path

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

mpirun -np 16 --host $host1:8,$host2:8 \
    $binary_path \
    -i "$train_data_path" \
    -j "$val_data_path" \
    -o $out_dir \
    -v 250 -s 20000 -g 144 \
    -h 1 \
    -b 64 -t 1024 \
    -d 2097152 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.1 \
    -u 700 \
    -n 1000 \
    -y 0 \
    -e d12 \
    -pi "mpi" \
