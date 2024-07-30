#!/bin/bash

# Downloads the FineWeb-Edu 100B dataset, but in an already tokenized format in .bin files
# Example: ./edu_fineweb.sh 100
# would download 100 shards
# Default is all shards
# Make sure to run this from current directory, i.e. inside ./dev/data!

# Check if MAX_SHARDS is provided as positional first arg, otherwise default to 1024
if [ $# -eq 0 ]; then
    MAX_SHARDS=1001
else
    MAX_SHARDS=$1
fi

if [ $MAX_SHARDS -gt 1001 ]; then
    MAX_SHARDS=1001
fi

# Base URLs
TRAIN_BASE_URL="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards/resolve/main/edu_fineweb_train_"
VAL_URL="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards/resolve/main/edu_fineweb_val_000000.bin"

# Directory to save files
SAVE_DIR="edu_fineweb100B"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

download() {
    local FILE_URL=$1
    local FILE_NAME=$(basename $FILE_URL | cut -d'?' -f1)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"
    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_NAME to $SAVE_DIR"
}

# Function to manage parallel jobs
run_in_parallel() {
    local max_jobs=$1
    shift
    local commands=("$@")
    local job_count=0

    for cmd in "${commands[@]}"; do
        eval "$cmd" &
        ((job_count++))
        if (( job_count >= max_jobs )); then
            wait -n
            ((job_count--))
        fi
    done

    # Wait for any remaining jobs to finish
    wait
}

# Export the function so it's available in subshells
export -f download

# Download the validation shard
download "$VAL_URL" &

# Generate train file shard download commands
train_commands=()
for i in $(seq -f "%06g" 1 $MAX_SHARDS); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    train_commands+=("download \"$FILE_URL\"")
done

# Run the train file commands in parallel
run_in_parallel 40 "${train_commands[@]}"
echo "The val shard and first $MAX_SHARDS train shards of FineWebEdu100B files downloaded in $SAVE_DIR"
