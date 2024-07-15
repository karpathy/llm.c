# GPT-2 (350M) repro on FineWeb
# 350M parameter model on ~30B tokens
# => 6 * 350e6 * 31.5e9 = 6.615e19 ~= 7e19 capability model (10X 124M)
# 60K steps on 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~820ms/iter
# => training time 60,000 steps * 820ms = 13.7 hours ~= $200 (10X 124M)

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt2_350M"
done_file="$out_dir/DONE_00060000"

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/fineweb.py --version 100B to prepro data
    # run python dev/data/hellaswag.py to prepro hellaswag eval
    mpirun -np 8 ./train_gpt2cu \
                -i "dev/data/fineweb100B/fineweb_train_*.bin" \
                -j "dev/data/fineweb100B/fineweb_val_*.bin" \
                -o $out_dir \
                -v 250 -s 100000 -g 144 \
                -h 1 \
                -b 64 -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0003 \
                -q 0.0 \
                -u 700 \
                -n 2000 \
                -x 60000 \
                -y 1 \
                -e "d24"

    sleep 1
done
