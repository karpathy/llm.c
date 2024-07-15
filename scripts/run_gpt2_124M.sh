# GPT-2 (124M) repro on FineWeb
# 124M parameter model on 10B tokens
# => 6 * 124e6 * 10e9 = 7.44e18 ~= 7e18 capability model
# 18,865 steps of 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~300ms/iter
# => training time 18,865 * 300ms = 94.3 min ~= $20

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt2_124M"
done_file="$out_dir/DONE_00018865"

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/fineweb.py --version 10B to prepro data
    # run python dev/data/hellaswag.py to prepro hellaswag eval
    mpirun -np 8 ./train_gpt2cu \
                -i "dev/data/fineweb10B/fineweb_train_*.bin" \
                -j "dev/data/fineweb10B/fineweb_val_*.bin" \
                -o $out_dir \
                -v 250 -s 20000 -g 144 \
                -h 1 \
                -b 64 -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.0 \
                -u 700 \
                -n 5000 \
                -y 1 \
                -e "d12"

    sleep 1
done
