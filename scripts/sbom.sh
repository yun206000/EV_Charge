#!/usr/bin/env bash
# SBOM (software bill of materials) of the service image with syft (Anchore, https://github.com/anchore/syft),
# README_for_doc.md section 11.12: Debian packages, Python packages and binaries of the image, SPDX 2.3 JSON.
# The syft release is downloaded once from GitHub and checked against its checksums file; the scan runs locally on the
# Docker image and sends nothing (update check off).
#
#   scripts/sbom.sh                       # IMAGE=pcrl:2.0-model OUT=Results/Security/充電站選址服務_SBOM.json
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-pcrl:2.0-model}
OUT=${OUT:-Results/Security/充電站選址服務_SBOM.json}
SYFT_VERSION=1.57.0
SYFT_DIR=${SYFT_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/syft-$SYFT_VERSION}
export SYFT_CHECK_FOR_APP_UPDATE=false SYFT_FORMAT_PRETTY=true

if [ ! -x "$SYFT_DIR/syft" ]; then
    case "$(uname -m)" in
        x86_64) arch=amd64 ;;
        aarch64 | arm64) arch=arm64 ;;
        *) echo "no syft release for $(uname -m)" >&2; exit 1 ;;
    esac
    archive=syft_${SYFT_VERSION}_linux_$arch.tar.gz
    url=https://github.com/anchore/syft/releases/download/v$SYFT_VERSION
    mkdir -p "$SYFT_DIR"
    curl -sSfL -o "$SYFT_DIR/$archive" "$url/$archive"
    curl -sSfL -o "$SYFT_DIR/checksums.txt" "$url/syft_${SYFT_VERSION}_checksums.txt"
    (cd "$SYFT_DIR" && grep " $archive\$" checksums.txt | sha256sum -c --quiet - && tar xzf "$archive" syft)
fi

mkdir -p "$(dirname "$OUT")"
"$SYFT_DIR/syft" scan "docker:$IMAGE" -q -o "spdx-json=$OUT"
echo "$OUT ($IMAGE, syft $SYFT_VERSION)"
