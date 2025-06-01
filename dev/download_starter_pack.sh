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

# Some colors
YELLOW='\033[1;33m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Sanity check
REQUIREMENTS=(
    "curl"
)
for requirement in ${REQUIREMENTS[@]}; do
  if ! command -v "$requirement" &> /dev/null
  then
      echo -e "${RED}Error: \"$requirement\" is required but not installed or not found in your PATH. Please install it and try again.${NC}"
      exit 1
  fi
done

# Function to get the current cursor position
cursor_row=0
cursor_col=0
files_to_download=${#FILES[@]}
get_cursor_position() {
    local pos
    # Send escape sequence to get the cursor position
    exec < /dev/tty
    oldstty=$(stty -g)
    stty raw -echo min 0
    echo -en "\033[6n" > /dev/tty

    # Read the response
    IFS=';' read -r -d R -a pos

    # Restore terminal settings
    stty $oldstty

    # Extract row and column
    cursor_row=$((${pos[0]:2}))
    cursor_col=$((${pos[1]}))
}

# Allocate space in the terminal for the messages
cursor_bottom_row=0
for file in "${FILES[@]}"; do
  echo ""
  get_cursor_position
done
cursor_bottom_row=$cursor_row

move_cursor() {
    local row=$1
    local col=$2
    echo -ne "\033[${row};${col}H"
}

# Function to download files to the appropriate directory
download_file() {
    local FILE_NAME=$1
    local ORDER=$2
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
    move_cursor $((ORDER+cursor_bottom_row-files_to_download-1)) 0
    echo -e "${YELLOW}Downloading $FILE_NAME...${NC}"
    if curl -s -L -o "$FILE_PATH" "$FILE_URL"; then
        move_cursor $((ORDER+cursor_bottom_row-files_to_download-1)) 0
        echo -e "${GREEN}Downloaded $FILE_NAME to $FILE_PATH${NC}   "
    else
        move_cursor $((ORDER+cursor_bottom_row-files_to_download-1)) 0
        echo -e "${RED}Failed to download $FILE_NAME${NC}   "
    fi
}

# Export the function so it's available in subshells
export -f download_file

# Function to handle SIGINT
declare -a pids
cleanup() {
    echo -e "${RED}Caught SIGINT signal! Terminating background processes...${NC}"
    for pid in "${pids[@]}"; do
        pkill -P "$pid" &>/dev/null
        kill -9 "$pid" &>/dev/null
    done
    exit 1
}

# Generate download commands
download_commands=()
order=1
for FILE in "${FILES[@]}"; do
    download_commands+=("download_file \"$FILE\" $((order))")
    order=$((order+1))
done

# Function to manage parallel jobs in increments of a given size
run_in_parallel() {
    local batch_size=$1
    shift
    local i=0
    local q=0
    local command

    for command; do
        eval "$command" &
        pids[$q]=$!
        ((i = (i + 1) % batch_size))
        q=$((q + 1))
        if [ "$i" -eq 0 ]; then
            wait
        fi
    done

    # Wait for any remaining jobs to finish
    wait
}

trap cleanup SIGINT

# Get the starting cursor row position
get_cursor_position

# Run the download commands in parallel in batches of 2
run_in_parallel 6 "${download_commands[@]}"

move_cursor $((cursor_bottom_row)) 0
echo "All files downloaded and saved in their respective directories"
