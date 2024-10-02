ARG XFORMERS_COMPILE_JOBS=128
ARG HIP_ARCHITECTURES="gfx90a gfx942"

FROM quay.io/pypa/manylinux_2_28_x86_64 as rocm

RUN set -ex && \
  usermod -a -G render,video $(whoami) && \
  dnf -y install https://www.elrepo.org/elrepo-release-8.el8.elrepo.noarch.rpm && \
  dnf config-manager --set-enabled elrepo-kernel && \
  dnf -y install https://repo.radeon.com/amdgpu-install/6.2.2/rhel/8.10/amdgpu-install-6.2.60202-1.el8.noarch.rpm

RUN set -ex && \
  dnf -y install amdgpu-dkms rocm 

RUN set -ex && \
  python3.11 -m pip install uv && \
  uv venv --python 3.11 && \
  source .venv/bin/activate

RUN set -ex && \
  cd /opt && \
  git clone --recursive https://github.com/rocm/xformers && \
  cd xformers && \
  git log -1 

RUN set -ex && \
  cd /opt/xformers && \
  uv pip install ninja && \
  uv pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/nightly/rocm6.2 && \
  uv pip install -r requirements-test.txt && \
  uv pip install -r requirements-benchmark.txt && \
  uv pip list

ARG XFORMERS_COMPILE_JOBS
ENV MAX_JOBS=${XFORMERS_COMPILE_JOBS} 
ARG HIP_ARCHITECTURES
ENV HIP_ARCHITECTURES=${HIP_ARCHITECTURES} 
RUN set -ex && \
  cd /opt/xformers && \
  uv build . --wheel --no-build-isolation --verbose --offline && \
  uv pip install dist/*.whl && \
  cd / && \
  uv run -- python -m xformers.info

ENV PATH="/.venv/bin:${PATH}"