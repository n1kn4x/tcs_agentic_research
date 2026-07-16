FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
        numpy \
        pandas \
        scipy \
        sympy \
        matplotlib \
        seaborn \
        scikit-learn \
        statsmodels \
        networkx

RUN useradd --uid 1000 --create-home experiment \
    && mkdir -p /workspace \
    && chown -R 1000:1000 /workspace

WORKDIR /workspace
USER 1000:1000

CMD ["sleep", "infinity"]
