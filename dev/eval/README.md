# eleuther eval readme

The goal here is to run the Eleuther Eval harness exactly in the same way as that used in the [huggingface LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).

The starting point is a `.bin` file trained by llm.c. We now have to export it to a huggingface model and then evaluate it.

To export the model, use [export_hf.py](export_hf.py). See its documentation up top. Eample usage, from this directory:

```bash
cd dev/eval
python export_hf.py --input model.bin --output output_dir
```

Where you point to your model .bin file, and huggingface files get written to output_dir. The script can optionally also upload to huggingface hub. One more post-processing that is advisable is to go into the `output_dir`, open up the `config.json` there and add one more entry into the json object:

```
"_attn_implementation": "flash_attention_2"
```

To use FlashAttention 2. We had trouble evaluating in bfloat16 without using FlashAttention 2 (the scores are much lower, and this was never fully resolved). This is a temporary hack/workaround.

Now that we have the model in huggingface format, we download the Eleuther Eval Harness repo and run it. Head over to the parent/root directory of the llm.c repo and:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness/
cd lm-evaluation-harness
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
pip install -e .
```

And then run the run_eval.sh script:

```bash
./dev/eval/run_eval.sh output_dir result_dir
```

Where output_dir can either be local output dir (above), or a huggingface repo name.This will write eval json objects to `./lm-evaluation-harness/results/results_dir`. It will print the results into console, e.g. for a 774M model we see:

```
----------------------------------------
arc_challenge_25shot.json      : 30.4608
gsm8k_5shot.json               : 0.1516
hellaswag_10shot.json          : 57.8072
mmlu_5shot.json                : 25.8682
truthfulqa_0shot.json          : 35.7830
winogrande_5shot.json          : 59.3528
----------------------------------------
Average Score                  : 34.9039
```

But you can additionally get these results later by running `summarize_eval.py`:

```bash
python dev/eval/summarize_eval.py lm-evaluation-harness/results/results_dir
```

The same information will be printed again.

For some reason, the evaluation is quite expensive and runs for somewhere around 1-3 hours, even though it should be a few minutes at most. This has not been satisfyingly resolved so far.