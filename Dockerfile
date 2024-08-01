# Specify the correct NVIDIA CUDA image with CUDNN and development tools
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install system packages
RUN apt update && apt install -y \
    git \
    wget

# Install Miniconda
RUN mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm ~/miniconda3/miniconda.sh \
    && ~/miniconda3/bin/conda init bash

# Setting the PATH environment variable for conda
ENV PATH /root/miniconda3/bin:$PATH


# Clone necessary repositories to the home directory
RUN git clone https://github.com/NVIDIA/cudnn-frontend.git ~/cudnn-frontend \
    && git clone https://github.com/karpathy/llm.c.git ~/llm.c
