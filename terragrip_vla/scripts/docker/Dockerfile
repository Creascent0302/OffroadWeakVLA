# TerraGrip-VLA on 8x RTX 5090 (Blackwell, sm_120).
#
# The base image already carries torch built against CUDA 12.8, which is the
# first toolkit that emits sm_120 code. Do not `apt install nvidia-cuda-toolkit`
# on top of this: that pulls CUDA 12.0, which cannot target Blackwell.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/workspace/.torch \
    HF_HOME=/workspace/.hf

# libgl/libglib are needed by opencv even in the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch/torchvision come from the base image; installing them again from PyPI
# would silently replace the cu128 build with a cu12x default one.
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision)\b' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

COPY . /workspace/terragrip_vla
WORKDIR /workspace/terragrip_vla

# Fails loudly if the image cannot actually run a Blackwell kernel.
RUN python scripts/env_check.py --allow-cpu

CMD ["bash"]
