import os
import time
import argparse
import wandb

# Please follow https://docs.wandb.ai/quickstart and setup wandb server and python client

logging_metrics = {
    'eval'      : 'Hellaswag Evaluation',
    'tel'       : 'Validation Loss',
    'trl'       : 'Training Loss',
    'norm'      : 'Normalized Gradient',
    'lr'        : 'Learning Rate',
    'latency'   : 'Latency Per Step (ms)',    
    'mfu'       : 'A100 FP16 MFU %',
    'tokens'    : 'Tokens Per Second'
}

def parse_line(line):
    for metric, header in logging_metrics.items():
        if metric in line:
            parts = line.split()
            step = int(parts[0].split(":")[1])
            value = float(parts[1].split(":")[1])
            return {header: value}, step

def wait_for_file(file_path):
    while not os.path.exists(file_path):
        print(f"Waiting for file {file_path} to be created...")
        time.sleep(5)
    return open(file_path, 'r')

def process_log_file(log_file_path):
    f = wait_for_file(log_file_path)
    while True:
        line = f.readline()
        if line:
            data_dict, step = parse_line(line.strip())
            wandb.log(data_dict, step=step)            
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wandb Logger")
    parser.add_argument("-l", "--log_dir", type=str, required=True, help="log directory of the training script")
    args = parser.parse_args()
    log_dir =  args.log_dir                         # same as the output log dir (-o <string>) used in ./train_gpt2cu
    wandb.init(project="llmc-training", id=log_dir) # will use log_dir as the run_id in wandb

    process_log_file(log_dir + '/main.log')
