import runhouse as rh


def train_cloud():
    # This can be modified to use other instance types or other cloud providers.
    cluster = rh.cluster(name="rh-cuda-gpu", instance_type="A10G:1").up_if_not()
    env = rh.env(name="llm_c_train",
                 setup_cmds=[
                     "python llm.c/prepro_tinyshakespeare.py",
                     "python llm.c/train_gpt2.py",
                     "sudo apt-get update",
                     "sudo apt install clang -y --fix-missing"
                 ],
                 working_dir="./")  # rsync over the current git root
    # Note this is cached, so it will not run the installation commands again unless they change.
    env.to(cluster)
    cluster.run(["cd llm.c; make train_gpt2",
                 "OMP_NUM_THREADS=8 llm.c/train_gpt2"])


if __name__ == "__main__":
    train_cloud()

