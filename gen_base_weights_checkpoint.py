"""
A script to create base model weight files for gpt2 training from scratch.
- Makes use of GPT class and write_model function from train_gpt2.py.
- Base model weights are sourced from HuggingFace's transformer library.

Usage: python ./gen_base_weights_checkpoint.py [OPTION]
    --model_type [ gpt2 | gpt2-medium | gpt2-large | gpt2-xl ]
        defaults to gpt2

NOTE: This script supports creation of base weights for all model sizes but 
currently only 124M parameter model ("gpt2") is supported by train_gpt2.cu.
"""

if __name__ == "__main__":
    import argparse
    from train_gpt2 import GPT, write_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt2", help="by default we generate weights for 124M param model")
    args = parser.parse_args()

    # only allow valid model types
    assert args.model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

    # get base model weights using the same method as GPT.from_pretrained() in train_gpt2.py
    model = GPT.from_pretrained(args.model_type)

    # mapping from model type to # of model params in millions
    model_to_param_mapping = {
        "gpt2": 124, 
        "gpt2-medium": 350,
        "gpt2-large": 774, 
        "gpt2-xl": 1558, 
    }

    # create filename using # of params
    filename = "gpt2_{}M_base".format((model_to_param_lookup[args.model_type]))

    # write out model weights in f32 and bf16 formats for use by train_gpt2.cu
    write_model(model, filename + ".bin", dtype="float32")
    write_model(model, filename + "_bf16.bin", dtype="bfloat16")