#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --input_bin FILE                  Input .bin to train on"
    echo "  --input_val_bin FILE              Input .bin to eval validation loss on"
    echo "  --output_dir DIR                  Output directory to write logs and checkpoints"
    echo "  --model NAME                      Model type: gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48"
    echo "  --batch_size SIZE                 Batch size"
    echo "  --sequence_length LENGTH          Sequence length"
    echo "  --total_batch_size SIZE           Total desired batch size"
    echo "  --num_iterations ITER             Number of iterations"
    echo "  --inference_only FLAG             Only run inference"
    echo "  --learning_rate RATE              Learning rate"
    echo "  --warmup_iters ITERS              Learning rate warmup iterations"
    echo "  --learning_rate_decay_frac FRAC   Learning rate decay fraction"
    echo "  --weight_decay DECAY              Weight decay"
    echo "  --grad_clip CLIP                  Maximum gradient magnitude"
    echo "  --val_loss_every STEPS            Steps between validation loss evaluation"
    echo "  --val_max_steps STEPS             Number of validation batches to average"
    echo "  --sample_every STEPS              Steps between model sampling"
    echo "  --overfit_single_batch FLAG       Overfit just one batch of data"
    echo "  --tensorcores FLAG                Use tensorcores"
    echo "  --device DEVICE                   Device to use"
    echo "  --compile FLAG                    Compile the model"
    echo "  --flash FLAG                      Use flash attention"
    echo "  --dtype TYPE                      Data type: float32|float16|bfloat16"
    echo "  --zero_stage STAGE                Zero redundancy optimizer stage"
    echo "  --write_tensors FLAG              Write tensors to disk"
    echo "  --slurm_timeout_hours HOURS       SLURM job timeout in hours"
    echo "  --slurm_cpus_per_task CPUS        Number of CPU cores per task"
    echo "  --slurm_mem_gb GB                 Memory per node in GB"
    echo "  --partition NAME                  SLURM partition to use"
    echo "  --testing FLAG                    Run in testing mode"
    echo "  --conda_env NAME                  Conda environment to activate"
}

# Default parameters
input_bin="dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
input_val_bin=""
output_dir=""
model="gpt2"
batch_size=4
sequence_length=64
total_batch_size=256
num_iterations=10
inference_only=0
learning_rate=1e-4
warmup_iters=0
learning_rate_decay_frac=1.0
weight_decay=0.0
grad_clip=1.0
val_loss_every=0
val_max_steps=20
sample_every=0
overfit_single_batch=1
tensorcores=0
device="cuda"
compile=0
flash=0
dtype="float32"
zero_stage=0
write_tensors=1
slurm_timeout_hours=4
slurm_cpus_per_task=10
slurm_mem_gb=4
partition="mlhiwi"
testing=false
conda_env="LLMs"

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_bin) input_bin="$2"; shift 2;;
        --input_val_bin) input_val_bin="$2"; shift 2;;
        --output_dir) output_dir="$2"; shift 2;;
        --model) model="$2"; shift 2;;
        --batch_size) batch_size="$2"; shift 2;;
        --sequence_length) sequence_length="$2"; shift 2;;
        --total_batch_size) total_batch_size="$2"; shift 2;;
        --num_iterations) num_iterations="$2"; shift 2;;
        --inference_only) inference_only="$2"; shift 2;;
        --learning_rate) learning_rate="$2"; shift 2;;
        --warmup_iters) warmup_iters="$2"; shift 2;;
        --learning_rate_decay_frac) learning_rate_decay_frac="$2"; shift 2;;
        --weight_decay) weight_decay="$2"; shift 2;;
        --grad_clip) grad_clip="$2"; shift 2;;
        --val_loss_every) val_loss_every="$2"; shift 2;;
        --val_max_steps) val_max_steps="$2"; shift 2;;
        --sample_every) sample_every="$2"; shift 2;;
        --overfit_single_batch) overfit_single_batch="$2"; shift 2;;
        --tensorcores) tensorcores="$2"; shift 2;;
        --device) device="$2"; shift 2;;
        --compile) compile="$2"; shift 2;;
        --flash) flash="$2"; shift 2;;
        --dtype) dtype="$2"; shift 2;;
        --zero_stage) zero_stage="$2"; shift 2;;
        --write_tensors) write_tensors="$2"; shift 2;;
        --slurm_timeout_hours) slurm_timeout_hours="$2"; shift 2;;
        --slurm_cpus_per_task) slurm_cpus_per_task="$2"; shift 2;;
        --slurm_mem_gb) slurm_mem_gb="$2"; shift 2;;
        --partition) partition="$2"; shift 2;;
        --testing) testing="$2"; shift 2;;
        --conda_env) conda_env="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown parameter: $1"; usage; exit 1;;
    esac
done

# Determine SLURM partition and maximum runtime
if [ "$partition" = "all" ]; then
    q="alldlc_gpu-rtx2080"
elif [ "$partition" = "ml" ]; then
    q="mldlc_gpu-rtx2080"
elif [ "$partition" = "mlhiwi" ]; then
    q="mlhiwidlc_gpu-rtx2080"
    echo "Using mlhiwi partition"
else
    echo "Unknown partition: $partition"
    usage
    exit 1
fi

if [ "$testing" = true ]; then
    maximum_runtime=60
else
    if [ "$q" = "alldlc_gpu-rtx2080" ] || [ "$q" = "mlhiwidlc_gpu-rtx2080" ]; then
        maximum_runtime=$((24 * 60 - 1))
    else
        maximum_runtime=$((24 * 60 * slurm_timeout_hours - 1))
    fi
fi

# Create a temporary script with the partition embedded
TMP_SCRIPT=$(mktemp /tmp/sbatch_script.XXXXXX)
cat <<EOF > $TMP_SCRIPT
#!/bin/bash
#SBATCH --job-name=gpt2_training
#SBATCH --output=slurm_logs/gpt2_training_%j.out
#SBATCH --error=slurm_logs/gpt2_training_%j.err
#SBATCH --time=$((maximum_runtime / 60)):00:00
#SBATCH --partition=$q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$slurm_cpus_per_task
#SBATCH --mem=${slurm_mem_gb}G

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Activate Conda environment if provided
if [ -n "$conda_env" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$conda_env"
fi

# Check if the required data file exists
if [ ! -f "$input_bin" ]; then
    echo "$input_bin not found. Generating data file..."
    python_file="dev/data/$(basename $(dirname "$input_bin")).py"
    if [ ! -f "$python_file" ]; then
        echo "Data generation script $python_file not found."
        exit 1
    else
        python "$python_file"
    fi
fi

echo "Running GPT2 training script..."

# Run the training script
python train_gpt2.py \
    --input_bin="$input_bin" \
    --input_val_bin="$input_val_bin" \
    --output_dir="$output_dir" \
    --model="$model" \
    --batch_size="$batch_size" \
    --sequence_length="$sequence_length" \
    --total_batch_size="$total_batch_size" \
    --num_iterations="$num_iterations" \
    --inference_only="$inference_only" \
    --learning_rate="$learning_rate" \
    --warmup_iters="$warmup_iters" \
    --learning_rate_decay_frac="$learning_rate_decay_frac" \
    --weight_decay="$weight_decay" \
    --grad_clip="$grad_clip" \
    --val_loss_every="$val_loss_every" \
    --val_max_steps="$val_max_steps" \
    --sample_every="$sample_every" \
    --overfit_single_batch="$overfit_single_batch" \
    --tensorcores="$tensorcores" \
    --device="$device" \
    --compile="$compile" \
    --flash="$flash" \
    --dtype="$dtype" \
    --zero_stage="$zero_stage" \
    --write_tensors="$write_tensors"
EOF

# Submit the temporary script
sbatch $TMP_SCRIPT

# Clean up the temporary script
rm $TMP_SCRIPT
