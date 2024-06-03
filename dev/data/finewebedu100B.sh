#!/bin/bash

# Base URLs
TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebEduTokenizedGPT2/resolve/main/edu_fineweb_train_"
VAL_URL="https://huggingface.co/datasets/chrisdryden/FineWebEduTokenizedGPT2/resolve/main/edu_fineweb_val_000000.bin?download=true"

# Directory to save files
SAVE_DIR="fineweb100B"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Function to download, decompress, and delete files
download() {
    local FILE_URL=$1
    local FILE_NAME=$(basename $FILE_URL | cut -d'?' -f1)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"

    # Download the file
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

# Download
download "$VAL_URL" &

# Generate train file commands
train_commands=()
for i in $(seq -f "%06g" 1 1028); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    train_commands+=("download \"$FILE_URL\"")
done

# Run the train file commands in parallel
run_in_parallel 40 "${train_commands[@]}"

echo "All files downloaded, decompressed, and cleaned up in $SAVE_DIR"
