# GPT-2 (1558M) repro on FineWeb
# 1558M parameter model on 100B tokens
# => 6 * 1558Me6 * 100e9 = 6.966e20 ~= 1e21 capability model
# => 100,000 steps on 1M tokens/step (1,048,576 to be precise)
# on 8X A100 80GB SXM ($14/hr) steps in ~7s/iter
# => training time 100,000 steps * 7s = 194.4 hours ~= 8.1 days ~= $2721.6

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt2_1558M"
done_file="$out_dir/DONE_00100000"

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
                -v 250 -s 300000 -g 144 \
                -h 1 \
                -b 16 -t 1024 \
                -d 1048576 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0002 \
                -q 0.1 \
                -u 700 \
                -n 2000 \
                -x 100000 \
                -y 1 \
                -e "d48"

    sleep 1
done
