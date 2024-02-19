FROM nvidia/cuda:11.4.3-devel-ubuntu20.04 AS python_base_cuda11.4

ENV DEBIAN_FRONTEND="noninteractive"

# Update system and install wget
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        curl \
        wget \
        libpython3.10 \
        npm \
        pandoc \
        ruby \
        software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Conda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > ~/miniconda.sh -s && \
    bash ~/miniconda.sh -b -p /home/user/conda && \
    rm ~/miniconda.sh
ENV PATH "/home/user/conda/bin:${PATH}"
RUN conda install python=3.10


#########################################################
## Development Env
#########################################################

FROM python_base_cuda11.4 as anomalib_development_env

RUN mkdir /app
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app/pytorch_models"