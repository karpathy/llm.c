"""
Script for running benchmarks on the Modal platform.
This is useful for folks who do not have access to expensive GPUs locally.

Example usage:
GPU_MEM=80 modal run benchmark_on_modal.py \
    --compile-command "nvcc -O3 --use_fast_math attention_forward.cu -o attention_forward -lcublas" \
    --run-command "./attention_forward 1"

This will mount the contents of the current directory to the remote container on modal,
compile the `attention_forward.cu` file with `nvcc`, and run the resulting binary on a A100 GPU with 80GB of memory.
"""

import subprocess
import os
import sys

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
GPU_CONFIG = GPU_NAME_TO_MODAL_CLASS_MAP[GPU_NAME](count=N_GPUS, size=str(GPU_MEM)+'GB')

APP_NAME = "llm.c benchmark run"

# We don't actually need to use the Axolotl image here, but it's reliable
AXOLOTL_REGISTRY_SHA = (
    "d5b941ba2293534c01c23202c8fc459fd2a169871fa5e6c45cb00f363d474b6a"
)
axolotl_image = (
    Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
    .run_commands(
        "git clone https://github.com/OpenAccess-AI-Collective/axolotl /root/axolotl",
        "cd /root/axolotl && git checkout v0.4.0",
    )
    .pip_install("huggingface_hub==0.20.3", "hf-transfer==0.1.5")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
        )
    )
)

stub = modal.App(APP_NAME)


def execute_command(command: str):
    command_args = command.split(" ")
    print(f"{command_args = }")
    subprocess.run(command_args, stdout=sys.stdout, stderr=subprocess.STDOUT)


@stub.function(
    gpu=GPU_CONFIG,
    image=axolotl_image,
    allow_concurrent_inputs=4,
    container_idle_timeout=900,
    # This copies everything in this folder to the remote root folder
    mounts=[modal.Mount.from_local_dir("./", remote_path="/root/")]
)
def run_benchmark(compile_command: str, run_command: str):
    execute_command("pwd")
    execute_command("ls")
    execute_command(compile_command)
    execute_command(run_command)
    return None


@stub.local_entrypoint()
def inference_main(compile_command: str, run_command: str):
    results = run_benchmark.remote(compile_command, run_command)
    return results
