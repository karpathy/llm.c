
# the same as run124M.sh but in PyTorch
# if you wish to train on just a single GPU, simply skip the torchrun part, i.e.
# python train_gpt2.py ... (all the other arguments the same)
torchrun --standalone --nproc_per_node=4 train_gpt2.py \
    --input_bin "dev/data/fineweb10B/fineweb_train_*.bin" \
    --output_dir pylog124M \
    --write_tensors 0 \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 18865 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0006 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.0 \
    --overfit_single_batch 0
