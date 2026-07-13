#!/bin/bash
# build-g10.sh — One-command build for G10
# Usage: ./scripts/build-g10.sh [phase]
#   phase=0 (default): vanilla colibri, no G10 flags
#   phase=1: add VRAM cache (COLI_G10_VRAM_CACHE)
#   phase=2: add tensor core decode
#   phase=3: full G10 build with all features

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/c"

PHASE="${1:-0}"

echo "=== Colibri-G10 Build: Phase $PHASE ==="

case "$PHASE" in
  0)
    echo "Phase 0: Vanilla colibri build"
    make clean
    make -j$(nproc)
    ;;
  1)
    echo "Phase 1: G10 VRAM Cache"
    make clean
    make G10=1 CUDA=1 -j$(nproc)
    ;;
  2)
    echo "Phase 2: G10 VRAM Cache + Tensor Core"
    make clean
    make G10=1 CUDA=1 G10_TENSOR_CORE=1 -j$(nproc)
    ;;
  3)
    echo "Phase 3: Full G10 Build"
    make clean
    make G10=1 CUDA=1 G10_TENSOR_CORE=1 G10_PROFILE=1 -j$(nproc)
    ;;
  *)
    echo "Unknown phase: $PHASE (valid: 0, 1, 2, 3)"
    exit 1
    ;;
esac

echo "=== Build complete ==="
ls -la glm
