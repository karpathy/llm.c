rm train_gpt2cu
make train_gpt2cu NO_MULTI_GPU=1

# Base model set arbitrarily here
base_width=256
base_num_heads=2
mup_base_attn_mult=$(echo "scale=3; sqrt($base_width / $base_num_heads)" | bc)
for me in 0 1; do  # 0 - SP; 1 - muP
    for mw in 64 128 256 512 1024 2048; do
        mm=$(echo "scale=3; $mw / $base_width" | bc)  # width multiplier
        echo "Running with muP=$me; width=$mw and width mult=$mm."
        ./train_gpt2cu \
            -i "dev/data/fineweb10B/fineweb_train_*.bin" \
            -j "dev/data/fineweb10B/fineweb_val_*.bin" \
            -b 32 -t 1024 \
            -d -1 \
            -r 0 \
            -z 0 \
            -l 0.006 \
            -y 0 \
            -me $me \
            -mm $mm \
            -ma $mup_base_attn_mult \
            -mc 1 \
            -mw $mw
    done
done
