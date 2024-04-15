# This script uses Runhouse and SkyPilot to launch, set up, and run the GPT-2 training
# script in the cloud. For installation instructions, see:
# https://www.run.house/docs/tutorials/quick-start-cloud
# After bringing up this box, you can ssh into it directly with `$ ssh rh-cuda-gpu`, or
# bring it down with `$ sky down rh-cuda-gpu`.

import runhouse as rh


def train_cloud():
    # This can be modified to use other instance types or other cloud providers.
    cluster = rh.cluster(name="rh-cuda-gpu", instance_type="A10G:1").up_if_not()
    env = rh.env(name="llm_c_train",
                 setup_cmds=[
                     "python -u llm.c/prepro_tinyshakespeare.py",
                     "python -u llm.c/train_gpt2.py",
                     "sudo apt-get update",
                     "sudo apt install clang -y --fix-missing"
                 ],
                 working_dir="./")  # rsync over the current git root and install requirements.txt
    # Note this is cached, so it will not run the installation commands again unless they change.
    env.to(cluster)
    cluster.run(["cd llm.c; make train_gpt2",
                 "OMP_NUM_THREADS=8 llm.c/train_gpt2"])


if __name__ == "__main__":
    train_cloud()

