#!/bin/bash

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Base URL
BASE_URL="https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/"

# Directory paths based on script location
SAVE_DIR_PARENT="$SCRIPT_DIR/.."
SAVE_DIR_TINY="$SCRIPT_DIR/data/tinyshakespeare"
SAVE_DIR_HELLA="$SCRIPT_DIR/data/hellaswag"

# Create the directories if they don't exist
mkdir -p "$SAVE_DIR_TINY"
mkdir -p "$SAVE_DIR_HELLA"

# Files to download
FILES=(
    "gpt2_124M.bin"
    "gpt2_124M_bf16.bin"
    "gpt2_124M_debug_state.bin"
    "gpt2_tokenizer.bin"
    "tiny_shakespeare_train.bin"
    "tiny_shakespeare_val.bin"
    "hellaswag_val.bin"
)

# Function to download files to the appropriate directory
download_file() {
    local FILE_NAME=$1
    local FILE_URL="${BASE_URL}${FILE_NAME}?download=true"
    local FILE_PATH

    # Determine the save directory based on the file name
    if [[ "$FILE_NAME" == tiny_shakespeare* ]]; then
        FILE_PATH="${SAVE_DIR_TINY}/${FILE_NAME}"
    elif [[ "$FILE_NAME" == hellaswag* ]]; then
        FILE_PATH="${SAVE_DIR_HELLA}/${FILE_NAME}"
    else
        FILE_PATH="${SAVE_DIR_PARENT}/${FILE_NAME}"
    fi

    # Download the file
    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_NAME to $FILE_PATH"
}

# Export the function so it's available in subshells
export -f download_file

# Generate download commands
download_commands=()
for FILE in "${FILES[@]}"; do
    download_commands+=("download_file \"$FILE\"")
done

# Function to manage parallel jobs in increments of a given size
run_in_parallel() {
    local batch_size=$1
    shift
    local i=0
    local command

    for command; do
        eval "$command" &
        ((i = (i + 1) % batch_size))
        if [ "$i" -eq 0 ]; then
            wait
        fi
    done

    # Wait for any remaining jobs to finish
    wait
}

# Run the download commands in parallel in batches of 2
run_in_parallel 6 "${download_commands[@]}"

echo "All files downloaded and saved in their respective directories"