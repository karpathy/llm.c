import submitit
import os
import subprocess
import argparse

def run_gpt2(args):
    # Check if the required data file exists
    data_file = args.input_bin
    if not os.path.exists(data_file):
        print(f"{data_file} not found. Generating data file...")
        python_file = "dev/data/" + data_file.split("/")[-2] + ".py"
        if not os.path.exists(python_file):
            raise FileNotFoundError(f"Data generation script {python_file} not found.")
        else:
            subprocess.run(["python", python_file], check=True)

    print("Running GPT2 training script...")

    # Run the training script
    subprocess.run([
        "python",
        "train_gpt2.py",
        f"--input_val_bin={args.input_val_bin}",
        f"--output_dir={args.output_dir}",
        f"--model={args.model}",
        f"--batch_size={args.batch_size}",
        f"--sequence_length={args.sequence_length}",
        f"--total_batch_size={args.total_batch_size}",
        f"--num_iterations={args.num_iterations}",
        f"--inference_only={args.inference_only}",
        f"--learning_rate={args.learning_rate}",
        f"--warmup_iters={args.warmup_iters}",
        f"--learning_rate_decay_frac={args.learning_rate_decay_frac}",
        f"--weight_decay={args.weight_decay}",
        f"--grad_clip={args.grad_clip}",
        f"--val_loss_every={args.val_loss_every}",
        f"--val_max_steps={args.val_max_steps}",
        f"--sample_every={args.sample_every}",
        f"--overfit_single_batch={args.overfit_single_batch}",
        f"--tensorcores={args.tensorcores}",
        f"--device={args.device}",
        f"--compile={args.compile}",
        f"--flash={args.flash}",
        f"--dtype={args.dtype}",
        f"--zero_stage={args.zero_stage}",
        f"--write_tensors={args.write_tensors}"
    ], check=True)

def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="Input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="Input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type: gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="Sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="Total desired batch size, in units of #tokens")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="Only run inference")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="Learning rate decay fraction")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Maximum gradient magnitude")
    parser.add_argument("--val_loss_every", type=int, default=0, help="Every how many steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="How many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="How often to sample from the model?")
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="Overfit just one batch of data")
    parser.add_argument("--tensorcores", type=int, default=0, help="Use tensorcores")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="Use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type: float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="Zero redundancy optimizer stage (0/1/2/3)")
    parser.add_argument("--write_tensors", type=int, default=1, help="Write tensors to disk")

    # SLURM specific arguments
    parser.add_argument("--slurm_timeout_hours", type=int, default=4, help="Timeout in hours")
    parser.add_argument("--slurm_cpus_per_task", type=int, default=10, help="Number of CPU cores per task")
    parser.add_argument("--slurm_mem_gb", type=int, default=4, help="Amount of memory per node in GB")

    parser.add_argument("-p", "--partition", default="mlhiwi", help="Slurm partition to use")
    parser.add_argument("-t", "--testing",type=bool, default=False, help="flag to run test")
    args = parser.parse_args()

    if args.partition == 'all':
        q = 'alldlc_gpu-rtx2080'
    if args.partition == 'ml':
        q = 'mldlc_gpu-rtx2080'
    if args.partition == 'mlhiwi':
        q = "mlhiwidlc_gpu-rtx2080"

    if args.testing:
        maximum_runtime = 60
    else:
        if q == 'alldlc_gpu-rtx2080' or q == 'mlhiwidlc_gpu-rtx2080':
            maximum_runtime = 24*60*1-1
        else:
            maximum_runtime = 24*60*args.slurm_timeout_hours-1

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
                        timeout_min=maximum_runtime,
                        slurm_partition=q, #  mldlc_gpu-rtx2080
                        slurm_signal_delay_s=180, # time to pass the USR2 signal to slurm before the job times out so that it can finish the run
                        # tasks_per_node=1,
                        # nodes=1,
                        cpus_per_task=args.slurm_cpus_per_task, #24
                        mem_gb=args.slurm_mem_gb,
                        # exclusive=False,
                        slurm_gres=f'gpu:{1}'
    )
    
    print("job to be submitted")

    job = executor.submit(run_gpt2, args)
    print(f"Submitted job_id: {job.job_id}")

if __name__ == "__main__":
    main()
