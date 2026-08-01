// REQUIRES: mpmath
// RUN: python3 %S/../../scripts/generate-q31-twiddle-tables.py

// Verify mode of the frozen-table generator: it re-derives every Q31 stage
// twiddle at 50 decimal digits, first reproducing the shipped Q15 authority
// (the keyed stage-2/4/8 sample and both complete extent-64 distinct word
// sets the compiler emits), then failing unless the committed header is the
// exact regeneration fixed point. Running it here keeps the committed
// Q31TwiddleTables.h pinned to its derivation instead of trusting a manual
// invocation.
