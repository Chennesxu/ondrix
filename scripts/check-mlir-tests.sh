#!/usr/bin/env bash
set -euo pipefail

fail=0

while IFS= read -r file; do
  if ! grep -q '^[[:space:]]*//[[:space:]]*RUN:' "${file}"; then
    echo "error: ${file} is missing a lit RUN line" >&2
    fail=1
  fi
done < <(find test -type f -name '*.mlir' | sort)

exit "${fail}"
