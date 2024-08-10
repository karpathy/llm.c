# GPT-2 (1558M) repro on FineWeb-EDU
# 1558M parameter model on 32B tokens
# => 6 * 1558e6 * 32e9 = 6.966e20 ~= 3e20 capability model
# 32,000 steps on ~1M tokens/step (1,048,576 to be precise)
# on 8X H100 80GB SXM ($28/hr) steps in 2.80s/iter
# => training time 32,000 steps * 2.7s => 24 hours ~= 1 day ~= $672

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt2_1558M"
done_file="$out_dir/DONE_00032000"

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    mpirun -np 8 ./train_gpt2cu \
                -i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
                -j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
                -o $out_dir \
                -v 250 -s 300000 -g 384 \
                -h 1 \
                -b 16 -t 1024 \
                -d 1048576 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -k "cosine" \
                -l 0.0006 \
                -q 0.1 \
                -u 700 \
                -n 2000 \
                -x 32000 \
                -ge 1 \
                -y 1 \
                -e "d48"

    sleep 1
done
