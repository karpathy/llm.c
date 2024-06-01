#!/bin/bash

# Base URLs
TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2Compressed/resolve/main/fineweb_train_"
VAL_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2Compressed/resolve/main/fineweb_val_000000.bin.tar.xz?download=true"

# Directory to save files
SAVE_DIR="fineweb100B"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Function to download, decompress, and delete files
download_and_decompress() {
    local FILE_URL=$1
    local FILE_NAME=$(basename $FILE_URL)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"

    # Download the file
    curl -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_NAME to $SAVE_DIR"

    # Decompress the file
    xz -d "$FILE_PATH"
    echo "Decompressed $FILE_NAME"

    # Extract the tar file
    tar -xf "${FILE_PATH%.xz}" -C "$SAVE_DIR"
    echo "Extracted ${FILE_NAME%.xz}"

    # Remove the tar file
    rm "${FILE_PATH%.xz}"
    echo "Deleted ${FILE_NAME%.xz}"
}

# Download, decompress, and delete the single validation file
download_and_decompress "$VAL_URL"

# Loop to download and decompress train files from 000001 to 001024
for i in $(seq -f "%06g" 1 1024); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin.tar.xz?download=true"
    download_and_decompress "$FILE_URL"
done

echo "All files downloaded, decompressed, and cleaned up in $SAVE_DIR"