# Docker GPU Setup

## Problem

Rootless Docker (used on Ubuntu 24.04) doesn't support `deploy.resources.reservations.devices` for GPU passthrough. The standard Docker Compose GPU config doesn't work.

## Solution

Three changes required:

### 1. Host: NVIDIA container runtime config

```bash
sudo sed -i 's/#no-cgroups = false/no-cgroups = true/' /etc/nvidia-container-runtime/config.toml
sudo systemctl restart docker
```

Required for rootless Docker — tells the NVIDIA runtime to skip cgroup setup.

### 2. docker-compose.yml

```yaml
services:
  tei:
    runtime: nvidia                    # NOT deploy.resources.reservations.devices
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

### 3. Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04    # NOT python:3.12-slim

RUN pip install torch --index-url https://download.pytorch.org/whl/cu124
```

The CUDA base image provides the runtime libraries. `torch+cu124` provides PyTorch with CUDA 12.4 support.

## Verification

```bash
docker compose exec tei python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

Should output: `CUDA: True, Device: NVIDIA GeForce GTX 1650`

## Hardware

- GPU: NVIDIA GeForce GTX 1650, 4096 MiB
- Model: SEA-LION E5 Embedding 600M — uses ~2,244 MiB VRAM
