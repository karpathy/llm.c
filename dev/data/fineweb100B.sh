#!/bin/bash

# Base URLs
TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_train_"
VAL_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_val_000000.bin?download=true"

# Directory to save files
SAVE_DIR="fineweb100B"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Download the single validation file
VAL_FILE_NAME="fineweb_val_000000.bin"
curl -L -o "${SAVE_DIR}/${VAL_FILE_NAME}" "$VAL_URL"
echo "Downloaded ${VAL_FILE_NAME} to ${SAVE_DIR}"

# Loop to download train files from 000001 to 001024
for i in $(seq -f "%06g" 1 1024); do
    FILE_NAME="fineweb_train_${i}.bin"
    URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    
    # Download the file
    curl -L -o "${SAVE_DIR}/${FILE_NAME}" "$URL"
    
    echo "Downloaded ${FILE_NAME} to ${SAVE_DIR}"
done

echo "All files downloaded to ${SAVE_DIR}"