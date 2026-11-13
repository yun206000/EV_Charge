# PCRL, legacy environment: the conda environment "PCRL" (Python 3.8.20, stable-baselines3 1.5.0, torch 1.12.0,
# gym 0.21.0, osmnx 1.2.1) in which the models were trained and all reported results were produced. Kept only to
# reproduce them; its packages have known vulnerabilities (README_for_doc.md section 11.9), use docker/Dockerfile otherwise.
# Build (from the repository root; docker/legacy-py38/Dockerfile.dockerignore is used automatically):
#         docker build -f docker/legacy-py38/Dockerfile -t pcrl:1.0 .
# Run:    docker run --rm -it --env-file docker/.env -v "$PWD":/workspace/PCRL -u "$(id -u):$(id -g)" pcrl:1.0 python -m transfer.evaluate ...
FROM continuumio/miniconda3:24.9.2-0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

# the environment is created first so that code changes do not rebuild it
# 1. conda packages (python 3.8.20, scipy, scikit-learn, openblas, ...)
COPY docker/legacy-py38/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && rm /tmp/environment.yml

# 2. pip packages; gym 0.21.0 only builds with the old pip/setuptools of the host environment, without build isolation
COPY docker/legacy-py38/requirements-pcrl.txt /tmp/requirements-pcrl.txt
RUN /opt/conda/envs/PCRL/bin/python -m pip install --no-cache-dir pip==23.1.2 setuptools==65.5.0 wheel==0.38.4 \
    && /opt/conda/envs/PCRL/bin/python -m pip install --no-cache-dir --no-build-isolation -r /tmp/requirements-pcrl.txt \
    && rm /tmp/requirements-pcrl.txt

# every command runs inside the PCRL environment
ENV PATH=/opt/conda/envs/PCRL/bin:$PATH \
    CONDA_DEFAULT_ENV=PCRL

# unprivileged default user; `docker run -u "$(id -u):$(id -g)"` still overrides it for writing to a mounted repo
RUN useradd --create-home --uid 1000 --shell /bin/bash pcrl
WORKDIR /workspace/PCRL
RUN chown pcrl:pcrl /workspace/PCRL

# only code and input data, listed explicitly (no secrets); owned by root, read-only for the user, outputs are
# written to new folders or to the mounted repo
COPY *.py ./
COPY transfer/ transfer/
COPY analysis/ analysis/
COPY Preprocessing/ Preprocessing/
COPY Visualisierung/ Visualisierung/
COPY scripts/ scripts/
COPY Graph/ Graph/
USER pcrl

CMD ["python", "-c", "import torch, stable_baselines3, gym, osmnx; print('PCRL environment OK: torch', torch.__version__, '| sb3', stable_baselines3.__version__, '| gym', gym.__version__, '| osmnx', osmnx.__version__)"]
