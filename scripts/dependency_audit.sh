#!/usr/bin/env bash
# Dependency vulnerability scan of the PCRL environment (README_for_doc.md section 11.9), run before the SonarQube scan:
#   1. the Python packages installed in the Docker image (pip list of pcrl:2.0) plus docker/requirements-test.txt;
#      local version labels are dropped (torch 2.15.0+cpu is audited as torch 2.15.0, the same release on PyPI)
#   2. pip-audit (PyPA) in a python:3.10 container against the PyPI / OSV advisory database -> Results/Security/pip_audit.json
#   3. scripts/dependency_audit.py: severity from OSV, CSV, summary and dependency-audit-sonar.json for SonarQube
# Only package names and versions are sent to PyPI / OSV.
#
#   scripts/dependency_audit.sh                 # IMAGE=pcrl:2.0 PYTHON=$HOME/anaconda3/envs/PCRL-py312/bin/python
#   IMAGE=pcrl:1.0 PYTHON=$HOME/anaconda3/envs/PCRL/bin/python OUT=Results/Security/legacy-py38 \
#       TEST_REQUIREMENTS=docker/legacy-py38/requirements-test.txt \
#       SONAR=Results/Security/legacy-py38/sonar.json scripts/dependency_audit.sh   # legacy environment, not imported
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-pcrl:2.0}
PYTHON=${PYTHON:-$HOME/anaconda3/envs/PCRL-py312/bin/python}
PIP_AUDIT_VERSION=2.9.0
OUT=${OUT:-Results/Security}
SONAR=${SONAR:-dependency-audit-sonar.json}
TEST_REQUIREMENTS=${TEST_REQUIREMENTS:-docker/requirements-test.txt}
mkdir -p "$OUT"

docker run --rm "$IMAGE" python -m pip list --format=freeze 2>/dev/null > "$OUT/installed_packages.txt"
grep -hv '^#' "$OUT/installed_packages.txt" "$TEST_REQUIREMENTS" | sed 's/ *#.*//; s/+[A-Za-z0-9.]*$//' | grep . \
    | sort -fu \
    > "$OUT/audited_packages.txt"
echo "$(wc -l < "$OUT/audited_packages.txt") packages ($IMAGE + $TEST_REQUIREMENTS)"

# exit code 1 of pip-audit means "vulnerabilities found"; the findings are rated and reported in step 3
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/$OUT":/audit python:3.10 sh -c \
    "pip install -q --user --disable-pip-version-check --no-warn-script-location pip-audit==$PIP_AUDIT_VERSION && \
     /tmp/.local/bin/pip-audit --no-deps --disable-pip --progress-spinner off -r /audit/audited_packages.txt \
         -f json -o /audit/pip_audit.json" || [ $? -eq 1 ]

"$PYTHON" scripts/dependency_audit.py --audit "$OUT/pip_audit.json" --out-dir "$OUT" --sonar "$SONAR"
