#!/bin/bash
# Pre-commit checks for this repository: the three things I have got wrong at least once.
#
# 1. Blocking rules, via the default run. The *exit code* is the verdict -- piping the output
#    into another program discards it, which is how an R-043 violation was committed once.
# 2. The three C-class counters must match docs/rules/size-thresholds.md. They only appear
#    under --report, but a CI test asserts they agree, so adding any .py can turn CI red.
# 3. Whitespace on the commit being pushed: CI runs `git show --check` at its compile step.
#
# Usage:  bash tools/precommit_check.sh   (exit 0 = safe to commit)
# Pre-commit discipline: the three checks I have now got wrong at least once each.
set -o pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE" || exit 1
fail=0
echo "=== 1. blocking rules (default run, exit code is the verdict) ==="
.venv/bin/python tools/check_conventions.py --quiet > /tmp/cc.out 2>&1
rc=$?
echo "   exit=$rc"; tail -2 /tmp/cc.out
[ $rc -ne 0 ] && fail=1
echo "=== 2. C-class counters must match size-thresholds.md ==="
.venv/bin/python -m pytest tests/test_check_conventions.py::test_size_report_counts_match_the_checker -q 2>&1 | tail -2
echo "=== 3. whitespace on the commit being pushed ==="
git show --check --format= HEAD 2>&1 | head -3
echo "(empty above = clean)"
exit $fail
