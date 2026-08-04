#!/usr/bin/env bash
set -euo pipefail

fail=0

while IFS= read -r file; do
  if ! grep -q '^[[:space:]]*//[[:space:]]*RUN:' "${file}"; then
    echo "error: ${file} is missing a lit RUN line" >&2
    fail=1
  fi
done < <(find test -type f -name '*.mlir' | sort)

while IFS= read -r file; do
  if ! grep -q '^[[:space:]]*;[[:space:]]*RUN:' "${file}"; then
    echo "error: ${file} is missing a lit RUN line" >&2
    fail=1
  fi
done < <(find test -type f -name '*.ll' -not -path 'test/*/Inputs/*' | sort)

# A test lit collects but Git does not carry is evidence that exists only on
# one machine. The .ll suffix is the live example: an ignore rule meant for
# build output silently swallowed a whole characterization suite.
untracked=$(git ls-files --others --ignored --exclude-standard -- test \
            | grep -E '\.(mlir|ll)$' || true)
untracked="${untracked}$(git ls-files --others --exclude-standard -- test | grep -E '\.(mlir|ll)$' || true)"
if [ -n "${untracked}" ]; then
  echo "error: these tests run locally but are not tracked:" >&2
  echo "${untracked}" >&2
  fail=1
fi

exit "${fail}"
