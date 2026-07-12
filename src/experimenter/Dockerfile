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

RUN useradd -m -u 1000 experimenter \
    && mkdir -p /workspace \
    && chown -R experimenter:experimenter /workspace

WORKDIR /workspace
USER experimenter

CMD ["sleep", "infinity"]
