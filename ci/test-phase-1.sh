#!/bin/bash
# test-phase-1.sh — Phase 1 gate: VRAM cache builds, CUDA backend tests pass, smoke test
set -euo pipefail

echo "=== Phase 1 Gate: G10 VRAM Cache ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/c"

# Step 1: Build with G10=1 CUDA=1
echo ""
echo "--- Step 1: Build with G10=1 CUDA=1 ---"
make clean
make G10=1 CUDA=1 CUDA_HOME=/usr/local/cuda-13.0 -j$(nproc) 2>&1
echo "BUILD: OK"

# Step 2: Verify the G10 flag compiled in
echo ""
echo "--- Step 2: Verify G10 symbols present ---"
nm glm | grep g10_vram_cache_init > /dev/null && echo "G10 symbols: OK" || { echo "FAIL: missing G10 symbols"; exit 1; }

# Step 3: Run standard C test suite
echo ""
echo "--- Step 3: Run standard test suite ---"
make test-c 2>&1
echo "STANDARD TESTS: OK"

# Step 4: Run CUDA backend correctness test
echo ""
echo "--- Step 4: Run CUDA backend test ---"
make cuda-test CUDA=1 CUDA_HOME=/usr/local/cuda-13.0 2>&1
echo "CUDA TESTS: OK"

# Step 5: Run CUDA benchmark (non-blocking)
echo ""
echo "--- Step 5: Run CUDA benchmark ---"
make cuda-bench CUDA=1 CUDA_HOME=/usr/local/cuda-13.0 2>&1
echo "CUDA BENCH: OK"

echo ""
echo "=== Phase 1 Gate: PASS ==="
