# https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
# (See About tab -> REPRODUCIBILITY)

# This script is intended to be run from the parent/root directory of llm.c repo.

# Clone the evaluation harness:

# git clone https://github.com/EleutherAI/lm-evaluation-harness/
# cd lm-evaluation-harness
# git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
# pip install -e .

# Then return to the parent directory and run this script

# cd ..
# ./dev/eval/run_eval.sh [model_name] [result_name]

# where model_name is either a HF model such as openai-community/gpt2 or a local path such as ./gpt2-124M-run1
# and result_name is the name of the folder under lm-evaluation-harness/results to store the evaluations

# Since the evals can take a couple of hours to run, depending on the model size, you may wish to
# run within a "screen" session or by using nohup to run the script:

# nohup ./dev/eval/run_eval.sh [model_name] [result_name] > run.txt 2> err.txt &

if [ -z "$1" ]; then
    echo "Error: missing HuggingFace model name or path to local model"
    echo "./run_eval.sh hf_account/model_name my_result"
  exit 1
fi
if [ -z "$2" ]; then
  echo "Error: missing output name for results"
    echo "./run_eval.sh hf_account/model_name my_result"
  exit 1
fi

export MODEL="$(realpath -s "$1")"
export RESULT="$2"
echo "Evaluating model $MODEL"
echo "Saving results to ./lm-evaluation-harness/results/$RESULT"

cd lm-evaluation-harness

python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True --tasks truthfulqa_mc --batch_size 1 --no_cache --write_out --output_path results/$RESULT/truthfulqa_0shot.json --device cuda
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True --tasks winogrande --batch_size 1 --no_cache --write_out --output_path results/$RESULT/winogrande_5shot.json --device cuda --num_fewshot 5
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True --tasks arc_challenge --batch_size 1 --no_cache --write_out --output_path results/$RESULT/arc_challenge_25shot.json --device cuda --num_fewshot 25
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True --tasks hellaswag --batch_size 1 --no_cache --write_out --output_path results/$RESULT/hellaswag_10shot.json --device cuda --num_fewshot 10
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True --tasks gsm8k --batch_size 1 --no_cache --write_out --output_path results/$RESULT/gsm8k_5shot.json --device cuda --num_fewshot 5
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions --batch_size 1 --no_cache --write_out --output_path results/$RESULT/mmlu_5shot.json --device cuda --num_fewshot 5

cd ..
python dev/eval/summarize_eval.py lm-evaluation-harness/results/$RESULT
