#!/bin/bash
# test-phase-0.sh — Baseline validation for Phase 0
# Tests: vanilla colibri builds, upstream tests pass, GPU path compiles
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/c"

echo "=== Phase 0: Baseline Validation ==="
PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    if "$@"; then
        echo "  ✅ $desc"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "--- Test 1: CPU build ---"
make clean 2>/dev/null
check "CPU build succeeds" make -j$(nproc) 2>&1

echo "--- Test 2: CPU test suite ---"
check "CPU tests pass (make test-c)" make test-c 2>&1

echo "--- Test 3: CUDA build compiles ---"
check "CUDA build succeeds" make clean 2>/dev/null '&&' make CUDA=1 -j$(nproc) 2>&1

echo "--- Test 4: Teacher-forcing (if glm_tiny model exists) ---"
if [ -f "glm_tiny/ref_glm.json" ]; then
    check "Teacher-forcing token match" SNAP=./glm_tiny ./glm 16 8 2 ref_glm.json 2>&1 | grep -q "Matching tokens: 20/20"
else
    echo "  ⚠️  glm_tiny model not found — skipping teacher-forcing test"
    echo "     Generate with: python3 tools/make_glm_tiny.py"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
