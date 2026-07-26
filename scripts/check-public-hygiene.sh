#!/usr/bin/env bash
set -euo pipefail

fail=0

if git ls-files | grep -E '(^|/)(reference|references|third_party_reference)(/|$)' >/dev/null; then
  echo "error: reference material directories must not be tracked" >&2
  fail=1
fi

if git ls-files | grep -E '(^|/)(benchmarks|benchmark-results|evaluation)(/|$)' >/dev/null; then
  echo "error: evaluation and benchmark artifacts belong in a separate repository" >&2
  fail=1
fi

if git ls-files | grep -E '\.(docx|xlsx|pdf)$' >/dev/null; then
  echo "error: binary reference/document files must not be tracked" >&2
  fail=1
fi

tracked_text_files=$(git grep -Il '' | grep -v '^scripts/check-public-hygiene.sh$' || true)
if [ -n "${tracked_text_files}" ]; then
  if git grep -n -E '/home/[^[:space:]]+|/Users/[^[:space:]]+|[A-Za-z]:\\' -- ${tracked_text_files}; then
    echo "error: tracked files contain local absolute paths" >&2
    fail=1
  fi
fi

exit "${fail}"
