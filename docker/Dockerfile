FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common build-essential git \
        python3-dev python3-pip wget curl cmake libeigen3-dev ffmpeg libsm6 libxext6

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH=/root/miniconda3/bin:$PATH
RUN conda init \
    && eval "$(conda shell.posix hook)" \
    && conda create -n diff3f python=3.10 && conda activate diff3f \
    && conda install -y -c conda-forge -c matplotlib numpy scikit-learn scipy meshplot jupyterlab jupyterlab_vim \
    && conda install -y pytorch=2.1.0 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia \
    && conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath \
    && conda install -y pytorch3d -c pytorch3d \
    && TORCH_CUDA_ARCH_LIST="5.0+PTX 6.0 6.1 7.0 7.5 8.0+PTX" pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers \
    && pip install --upgrade diffusers==0.21.4 einops huggingface-hub meshio opencv-python plyfile transformers trimesh potpourri3d robust_laplacian accelerate

RUN echo "source activate diff3f" > ~/.bashrc
ENV PATH /root/miniconda3/envs/diff3f/bin:$PATH

