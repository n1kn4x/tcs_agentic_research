FROM node:22-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        gfortran \
        git \
        jq \
        procps \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent

RUN python3 -m pip install \
        numpy \
        pandas \
        scipy \
        sympy \
        matplotlib \
        seaborn \
        scikit-learn \
        statsmodels \
        networkx \
        ipython \
        jupyter

# The upstream node image already defines UID 1000 as `node` on some tags.
# Use numeric UID/GID 1000 directly so the experimenter workspace stays writable
# on typical host bind mounts without assuming the username is available.
RUN mkdir -p /workspace \
    && chown -R 1000:1000 /workspace

WORKDIR /workspace
USER 1000:1000

CMD ["sleep", "infinity"]
