FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG DEBIAN_FRONTEND=noninteractive

# Core build + debug toolchain, Eigen, Python, pybind11
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    ca-certificates \
    curl \
    wget \
    gnupg \
    libeigen3-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    pybind11-dev \
    python3-pybind11 \
    gdb \
    clang \
    lldb \
    ripgrep \
  && rm -rf /var/lib/apt/lists/*

# Node.js 22 (NodeSource)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
  && apt-get update \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# Verify Node / npm
RUN node -v && npm -v

# Install global CLIs (optional but matches your workflow)
RUN npm install -g \
  @openai/codex@latest \
  @anthropic-ai/claude-code@latest \
  @google/gemini-cli@latest

# Ensure vscode home/cache are writable
RUN mkdir -p /home/vscode/.cache \
  && chown -R vscode:vscode /home/vscode

ENV SHELL=/bin/bash
WORKDIR /workspaces/CRM_DiffSim_Cl
USER vscode
CMD ["/bin/bash"]
