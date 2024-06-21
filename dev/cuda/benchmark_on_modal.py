"""
Script for running benchmarks on the Modal platform.
This is useful for folks who do not have access to expensive GPUs locally.
Example usage for cuda kernels:
GPU_MEM=80 modal run benchmark_on_modal.py \
    --compile-command "nvcc -O3 --use_fast_math attention_forward.cu -o attention_forward -lcublas" \
    --run-command "./attention_forward 1"
OR if you want to use cuDNN etc.


For training the gpt2 model with cuDNN use:
GPU_MEM=80 modal run dev/cuda/benchmark_on_modal.py \
    --compile-command "make train_gpt2cu USE_CUDNN=1"
    --run-command "./train_gpt2cu -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin -j dev/data/tinyshakespeare/tiny_shakespeare_val.bin -v 250 -s 250 -g 144 -f shakespeare.log -b 4"


For profiling using nsight system:
GPU_MEM=80 modal run dev/cuda/benchmark_on_modal.py \
    --compile-command "make train_gpt2cu USE_CUDNN=1" \
    --run-command "nsys profile --cuda-graph-trace=graph --python-backtrace=cuda --cuda-memory-usage=true \
    ./train_gpt2cu -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
    -j dev/data/tinyshakespeare/tiny_shakespeare_val.bin -v 250 -s 250 -g 144 -f shakespeare.log -b 4"

For more nsys profiling specifics and command options, take a look at: https://docs.nvidia.com/nsight-systems/2024.2/UserGuide/
-> To profile the report using a GUI, download NVIDIA NSight System GUI version (this software can run on all OS, so you download it locally)

NOTE: Currently there is a bug in the profiling using nsight system which produces a unrecognized GPU UUId error on the command line but it
does not actually interfere with the model training and validation. The report (that you download) is still generated and can be viewed from Nsight Systems
"""
import subprocess
import os
import sys
import datetime

import modal
from modal import Image, Stub
GPU_NAME_TO_MODAL_CLASS_MAP = {
    "H100": modal.gpu.H100,
    "A100": modal.gpu.A100,
    "A10G": modal.gpu.A10G,
}
N_GPUS = int(os.environ.get("N_GPUS", 1))
GPU_MEM = int(os.environ.get("GPU_MEM", 40))
GPU_NAME = os.environ.get("GPU_NAME", "A100")
GPU_CONFIG = GPU_NAME_TO_MODAL_CLASS_MAP[GPU_NAME](count=N_GPUS, size=str(GPU_MEM) + 'GB')

APP_NAME = "llm.c benchmark run"

image = (
    Image.from_registry("totallyvyom/cuda-env:latest-2")
    .pip_install("huggingface_hub==0.20.3", "hf-transfer==0.1.5")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
        )
    )
    .run_commands(
    "wget -q https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-Linux-x86_64.sh",
    "bash cmake-3.28.1-Linux-x86_64.sh --skip-license --prefix=/usr/local",
    "rm cmake-3.28.1-Linux-x86_64.sh",
    "ln -s /usr/local/bin/cmake /usr/bin/cmake",)
    .run_commands(
        "apt-get install -y --allow-change-held-packages libcudnn8 libcudnn8-dev",
        "apt-get install -y openmpi-bin openmpi-doc libopenmpi-dev kmod sudo",
        "git clone https://github.com/NVIDIA/cudnn-frontend.git /root/cudnn-frontend",
        "cd /root/cudnn-frontend && mkdir build && cd build && cmake .. && make"
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
        mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
        add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /\" && \
        apt-get update"
    ).run_commands(
        "apt-get install -y nsight-systems-2023.3.3"
    )
)

stub = modal.App(APP_NAME)

def execute_command(command: str):
    command_args = command.split(" ")
    print(f"{command_args = }")
    subprocess.run(command_args, stdout=sys.stdout, stderr=subprocess.STDOUT)

@stub.function(
    gpu=GPU_CONFIG,
    image=image,
    allow_concurrent_inputs=4,
    container_idle_timeout=900,
    mounts=[modal.Mount.from_local_dir("./", remote_path="/root/")],
    # Instead of 'cuda-env' put your volume name that you create from 'modal volume create {volume-name}'
    # This enables the profiling reports to be saved on the volume that you can download by using:
    # 'modal volume get {volume-name} {/output_file_name}
    # For example right now, when profiling using this command "nsys profile --trace=cuda,nvtx --cuda-graph-trace=graph --python-backtrace=cuda --cuda-memory-usage=true" you would get your report
    # using in a directory in your volume, where the name contains the timestamp unique id.
    # This script will generate a "report1_{timestamp} folder in volume"
    # and you can download it with 'modal volume get {volume-name} report1_{timestamp}
    volumes={"/cuda-env": modal.Volume.from_name("cuda-env")},
)
def run_benchmark(compile_command: str, run_command: str):
    execute_command("pwd")
    execute_command("ls")
    execute_command(compile_command)
    execute_command(run_command)
    # Use this section if you want to profile using nsight system and install the reports on your volume to be locally downloaded
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    execute_command("mkdir report1_" + timestamp)
    execute_command("mv /root/report1.nsys-rep /root/report1_" + timestamp + "/")
    execute_command("mv /root/report1.qdstrm /root/report1_" + timestamp + "/")
    execute_command("mv /root/report1_" + timestamp + "/" + " /cuda-env/")

    return None

@stub.local_entrypoint()
def inference_main(compile_command: str, run_command: str):
    results = run_benchmark.remote(compile_command, run_command)
    return results