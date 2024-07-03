ARG UBUNTU_VERSION=jammy
ARG PYTHON_VERSION=3.11
ARG PYTORCH_ROCM_ARCH=gfx942
ARG XFORMERS_URL=https://github.com/rocm/xformers
ARG XFORMERS_GIT_BRANCH=develop
ARG ROCM_VERSION=6.1.3
ARG XFORMERS_COMPILE_JOBS=64

FROM ubuntu:${UBUNTU_VERSION} as rocm
ENV DEBIAN_FRONTEND=noninteractive

ARG ROCM_VERSION
ENV ROCM_VERSION=${ROCM_VERSION}

RUN set -ex && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential git curl gpg gpg-agent ca-certificates

RUN set -ex && \
  mkdir --parents --mode=0755 /etc/apt/keyrings && \
  curl https://repo.radeon.com/rocm/rocm.gpg.key | \
    gpg -o /etc/apt/keyrings/rocm.gpg --dearmor && \
  . /etc/os-release && \
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/$ROCM_VERSION/ubuntu $UBUNTU_CODENAME main" > /etc/apt/sources.list.d/amdgpu.list && \
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION $UBUNTU_CODENAME main" > /etc/apt/sources.list.d/rocm.list && \
  apt-get update && \
  apt-get install -y \
    rocm-dev${ROCM_VERSION} rocm-llvm-dev${ROCM_VERSION} rocm-libs${ROCM_VERSION}

ENV ROCM_PATH="/opt/rocm"
ENV PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin":${PATH}

FROM rocm as conda
ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV CONDA_PREFIX="/opt/conda"
ENV CONDA_PYTHON=${CONDA_PREFIX}/envs/xformers/bin/python
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

RUN set -ex && \
  curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
  sha256sum miniconda.sh > miniconda.sha256 && \
  bash -exu miniconda.sh -bp ${CONDA_PREFIX} && \
  rm miniconda.sh && \
  conda init bash && \
  conda create -n xformers -y python=${PYTHON_VERSION} && \
  ${CONDA_PYTHON} -m pip install -U torch --index-url=https://download.pytorch.org/whl/nightly/rocm6.1 && \
  ${CONDA_PYTHON} -m pip install ninja pytest scipy

FROM conda as xformers
ARG XFORMERS_URL
ARG XFORMERS_GIT_BRANCH
ARG XFORMERS_COMPILE_JOBS
RUN set -ex && \
  MAX_JOBS=${XFORMERS_COMPILE_JOBS} ${CONDA_PYTHON} -m pip install git+${XFORMERS_URL}@${XFORMERS_GIT_BRANCH} --verbose
