#!/bin/bash

# Base URLs
TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_train_"
VAL_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_val_000000.bin?download=true"

# Directory to save files
SAVE_DIR="fineweb100B"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Function to download, decompress, and delete files
download_and_decompress() {
    local FILE_URL=$1
    local FILE_NAME=$(basename $FILE_URL | cut -d'?' -f1)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"

    # Download the file
    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_NAME to $SAVE_DIR"
}

# Function to manage parallel jobs in increments of a given size
run_in_parallel() {
    local batch_size=$1
    shift
    local commands=("$@")

    for ((i = 0; i < ${#commands[@]}; i += batch_size)); do
        for ((j = 0; j < batch_size && (i + j) < ${#commands[@]}; j++)); do
            eval "${commands[i + j]}" &
        done
        # Wait for the current batch of jobs to finish
    done
}

# Export the function so it's available in subshells
export -f download_and_decompress

# Download, decompress, and delete the single validation file in the background
download_and_decompress "$VAL_URL" &

# Generate train file commands
train_commands=()
for i in $(seq -f "%06g" 1 1024); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    train_commands+=("download_and_decompress \"$FILE_URL\"")
done

# Run the train file commands in parallel in batches of 8
run_in_parallel 40 "${train_commands[@]}"

echo "All files downloaded, decompressed, and cleaned up in $SAVE_DIR"
