# PCRL: Python 3.12.16, stable-baselines3 2.9.2, gymnasium 1.4.0, torch 2.15.0 (CPU), osmnx 2.1.1; packages without known
# vulnerabilities (README_for_doc.md sections 11.9, 11.10). The models were trained in the legacy environment
# docker/legacy-py38/Dockerfile and are loaded here unchanged (gym_compat.py).
# Build (from the repository root; docker/Dockerfile.dockerignore is used automatically):
#         docker build -f docker/Dockerfile -t pcrl:2.0 .
# Run:    docker run --rm -it --env-file docker/.env -v "$PWD":/workspace/PCRL -u "$(id -u):$(id -g)" pcrl:2.0 python -m transfer.evaluate ...
FROM python:3.12.16-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    PY=python

# the packages are installed first so that code changes do not reinstall them
COPY docker/requirements-pcrl.txt /tmp/requirements-pcrl.txt
RUN python -m pip install --no-cache-dir --root-user-action=ignore pip==26.3 setuptools==84.2.1 wheel==0.48.0 \
    && python -m pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements-pcrl.txt \
    && rm /tmp/requirements-pcrl.txt

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

CMD ["python", "-c", "import torch, stable_baselines3, gymnasium, osmnx; print('PCRL environment OK: torch', torch.__version__, '| sb3', stable_baselines3.__version__, '| gymnasium', gymnasium.__version__, '| osmnx', osmnx.__version__)"]
