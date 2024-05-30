# GPT-3 (124M) repro on FineWeb
# 124M parameter model on 300B tokens
# note context length: 1024 -> 2048 for GPT-3
# => 6 * 124e6 * 300e9 = 7.44e18 ~= 2.2e20 capability model
# 565,950 steps of 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~300ms/iter
# => training time 565,950 * 300ms ~= 47 hours ~= $658

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt3_124M"
done_file="$out_dir/DONE_00565950"

while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/fineweb.py --version 10B to prepro data
    # run python dev/data/hellaswag.py to prepro hellaswag eval
    mpirun -np 8 ./train_gpt2cu \
                -i "dev/data/fineweb100B/fineweb_train_*.bin" \
                -j "dev/data/fineweb100B/fineweb_val_*.bin" \
                -o $out_dir \
                -v 250 -s 20000 -g 144 \
                -h 1 \
                -b 32 -t 2048 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.0 \
                -u 700 \
                -n 10000 \
                -y 1 \
                -x 565950 \
                -e "d12"

    sleep 1
done
