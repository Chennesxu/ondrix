#!/usr/bin/env python3
"""Regenerate the frozen signed Q1.31 radix-2 stage twiddle tables.

The Q15 twiddle contract quantizes binary64 cos/sin estimates inside the
compiler and fails closed through a 2^-20 LSB tie guard
(include/ondrix/Support/GuardedQ15Quantization.h). That argument does not
transfer to Q31: 2^-20 Q31 LSB is 2^-51, about 4.4e-16 absolute, while a
binary64 cos evaluation already errs at about 1e-16. The guard would then be
only ~4x the evaluation error, so passing it would no longer prove that the
estimate and the real-valued definition round to the same integer. The Q31
tables are therefore evaluated OFFLINE at 50 decimal digits and checked in as
literal constants; the compiler reads the frozen table and never evaluates a
transcendental function for a Q31 twiddle.

The measured margins are far wider than the guard (worst 8.73e-02 LSB, i.e.
4.1e-11 absolute), so the tables are not marginal — what fails at Q31 is the
in-compiler *admissibility proof*, not the values themselves.

Usage (requires mpmath; not a build or CI dependency):

  python3 scripts/generate-q31-twiddle-tables.py            # verify only
  python3 scripts/generate-q31-twiddle-tables.py --write    # rewrite header

The self-check reproduces the Q15 authority the compiler already emits before
any Q31 value is trusted.
"""

import argparse
import pathlib
import sys

from mpmath import cos, floor, mp, mpf, pi, sin

mp.dps = 50

# Stage twiddles are needed for stage sizes 2, 4, ..., MAX_EXTENT, so the
# flat table is indexed by size/2 + index and slot 0 stays unused.
MAX_EXTENT = 64
HEADER = pathlib.Path(__file__).resolve().parent.parent / (
    "include/ondrix/Support/Q31TwiddleTables.h"
)

# Packed Q15 stage twiddles the compiler emits today for extent 8 in both
# directions, as the signed i32 constants pinned in
# test/Conversion/ondrix-to-ondsp/cfft_q15.mlir. They are the independent
# authority for this script's quantization rule: if the 50-digit path cannot
# reproduce the shipped Q15 table bit-exactly at Q15 scale, its Q31 output
# must not be trusted either.
Q15_AUTHORITY = {
    # (direction, size, index): packed word, imaginary << 16 | real
    ("forward", 2, 0): 32767,
    ("forward", 4, 0): 32767,
    ("forward", 4, 1): -2147483648,
    ("forward", 8, 0): 32767,
    ("forward", 8, 1): -1518445950,
    ("forward", 8, 2): -2147483648,
    ("forward", 8, 3): -1518426754,
    ("inverse", 2, 0): 32767,
    ("inverse", 4, 0): 32767,
    ("inverse", 4, 1): 2147418112,
    ("inverse", 8, 0): 32767,
    ("inverse", 8, 1): 1518492290,
    ("inverse", 8, 2): 2147418112,
    ("inverse", 8, 3): 1518511486,
}


def quantize(value, fractional_bits):
    """Round-half-even signed Q1.f quantization with declared saturation.

    Returns (integer, tie_distance_in_lsb, saturated). The tie distance is
    |fraction - 1/2| of the scaled value: the margin between this coefficient
    and the nearest rounding tie.
    """
    scale = mpf(2) ** fractional_bits
    scaled = value * scale
    lower = floor(scaled)
    fraction = scaled - lower
    distance = abs(fraction - mpf(1) / 2)
    if distance == 0:
        raise SystemExit("exact rounding tie: the frozen table would be ambiguous")
    quantized = int(lower) + (1 if fraction > mpf(1) / 2 else 0)
    minimum, maximum = -int(scale), int(scale) - 1
    saturated = quantized > maximum or quantized < minimum
    return max(minimum, min(maximum, quantized)), distance, saturated


def components(direction, size, index, fractional_bits):
    """W(size, index) = e^(-2*pi*i*index/size), conjugated for the inverse."""
    angle = 2 * pi * index / size
    sign = -1 if direction == "forward" else 1
    real, real_distance, real_saturated = quantize(cos(angle), fractional_bits)
    imaginary, imaginary_distance, imaginary_saturated = quantize(
        sign * sin(angle), fractional_bits
    )
    return (
        real,
        imaginary,
        min(real_distance, imaginary_distance),
        int(real_saturated) + int(imaginary_saturated),
    )


def table(direction, fractional_bits):
    """Flat stage table indexed by size/2 + index, slot 0 unused."""
    entries = [(0, 0)] * MAX_EXTENT
    worst = mpf("inf")
    saturations = 0
    size = 2
    while size <= MAX_EXTENT:
        for index in range(size // 2):
            real, imaginary, distance, saturated = components(
                direction, size, index, fractional_bits
            )
            entries[size // 2 + index] = (real, imaginary)
            worst = min(worst, distance)
            saturations += saturated
        size *= 2
    return entries, worst, saturations


def worst_margin_per_extent(fractional_bits):
    """Worst tie margin over exactly the stage twiddles one extent needs."""
    margins = {}
    for extent in (4, 8, 16, 32, 64):
        worst = mpf("inf")
        for direction in ("forward", "inverse"):
            size = 2
            while size <= extent:
                for index in range(size // 2):
                    worst = min(
                        worst, components(direction, size, index, fractional_bits)[2]
                    )
                size *= 2
        margins[extent] = worst
    return margins


def selfCheck():
    """Reproduce the shipped Q15 words before trusting the Q31 output."""
    for (direction, size, index), expected in sorted(Q15_AUTHORITY.items()):
        real, imaginary, _, _ = components(direction, size, index, 15)
        packed = ((imaginary & 0xFFFF) << 16) | (real & 0xFFFF)
        packed -= 1 << 32 if packed >= 1 << 31 else 0
        if packed != expected:
            print(
                f"Q15 self-check FAILED for {direction} W({size},{index}): "
                f"got {packed}, compiler emits {expected}",
                file=sys.stderr,
            )
            return False
    print(f"Q15 self-check: {len(Q15_AUTHORITY)} shipped packed words reproduced")
    return True


def render(forward, inverse, margins):
    lines = []
    lines.append("#ifndef ONDRIX_SUPPORT_Q31TWIDDLETABLES_H")
    lines.append("#define ONDRIX_SUPPORT_Q31TWIDDLETABLES_H")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("#include <optional>")
    lines.append("")
    lines.append("namespace ondrix {")
    lines.append("")
    lines.append("// GENERATED by scripts/generate-q31-twiddle-tables.py; do not edit by hand.")
    lines.append("//")
    lines.append("// Frozen signed Q1.31 radix-2 stage twiddles W(size, index) =")
    lines.append("// e^(-2*pi*i*index/size), quantized round-half-even with declared")
    lines.append("// saturation at +1.0. Unlike the Q15 twiddle and DCT tables, these values")
    lines.append("// are NOT recomputed in the compiler from binary64 cos/sin under the 2^-20")
    lines.append("// LSB tie guard. At Q31 that guard is only 2^-51 (4.4e-16) absolute while a")
    lines.append("// binary64 evaluation chain already errs at about 1e-16, so passing the")
    lines.append("// guard would no longer prove that the estimate rounds like the real-valued")
    lines.append("// definition. The tables are therefore evaluated offline at 50 decimal")
    lines.append("// digits and frozen here, and the conversion reads them directly.")
    lines.append("//")
    lines.append("// The values themselves are not marginal; worst tie margin per supported")
    lines.append("// extent, over both directions, in Q31 LSB units:")
    for extent, margin in margins.items():
        lines.append(f"//   N = {extent:2}: {float(margin):.6e}")
    lines.append("// The narrowest is about 9.1e+04 times the 2^-20 guard, so the frozen")
    lines.append("// integers are the correctly rounded ones by a wide margin. What fails at")
    lines.append("// Q31 is the in-compiler admissibility proof, not the coefficients.")
    lines.append("")
    lines.append("// Largest complex extent the frozen tables cover. Stage sizes 2, 4, ...,")
    lines.append("// kMaxQ31TwiddleExtent are tabulated, so any CFFT extent up to this bound")
    lines.append("// is served; a larger extent must fail closed until the tables grow.")
    lines.append(f"inline constexpr int64_t kMaxQ31TwiddleExtent = {MAX_EXTENT};")
    lines.append("")
    lines.append("struct Q31TwiddleComponents {")
    lines.append("  int32_t real;")
    lines.append("  int32_t imaginary;")
    lines.append("};")
    lines.append("")
    lines.append("// Forward and inverse are stored separately rather than derived from one")
    lines.append("// another: saturating quantization does not commute with negation at the")
    lines.append("// +-1.0 endpoints, where the forward imaginary part is exactly -2^31 but")
    lines.append("// its inverse counterpart saturates to 2^31 - 1.")
    lines.append("enum class Q31TwiddleDirection { Forward, Inverse };")
    lines.append("")
    lines.append("// Flat stage table indexed by size/2 + index, matching the recursive")
    lines.append("// combine's W(size, index) query; slot 0 is unused.")
    for name, entries in (("Forward", forward), ("Inverse", inverse)):
        lines.append(
            f"inline constexpr Q31TwiddleComponents k{name}Q31Twiddles[kMaxQ31TwiddleExtent] = {{"
        )
        for real, imaginary in entries:
            lines.append(f"    {{{real}, {imaginary}}},")
        lines.append("};")
        lines.append("")
    lines.append("// Packed Q31 complex word of one stage twiddle: imaginary in bits 63..32,")
    lines.append("// real in bits 31..0. Returns std::nullopt outside the frozen table so")
    lines.append("// every consumer fails closed instead of extrapolating.")
    # Emitted pre-formatted so the checked-in header is already
    # clang-format clean and the regeneration check below stays meaningful.
    lines.append(
        "inline std::optional<uint64_t> getPackedQ31TwiddleBits("
        "Q31TwiddleDirection direction, int64_t size,"
    )
    lines.append("                                                       int64_t index) {")
    lines.append("  if (size < 2 || size > kMaxQ31TwiddleExtent || index < 0 || index >= size / 2)")
    lines.append("    return std::nullopt;")
    lines.append("  const Q31TwiddleComponents &entry = direction == Q31TwiddleDirection::Forward")
    lines.append("                                          ? kForwardQ31Twiddles[size / 2 + index]")
    lines.append("                                          : kInverseQ31Twiddles[size / 2 + index];")
    lines.append("  return (static_cast<uint64_t>(static_cast<uint32_t>(entry.imaginary)) << 32) |")
    lines.append("         static_cast<uint64_t>(static_cast<uint32_t>(entry.real));")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace ondrix")
    lines.append("")
    lines.append("#endif // ONDRIX_SUPPORT_Q31TWIDDLETABLES_H")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="rewrite the generated header in place"
    )
    arguments = parser.parse_args()

    if not selfCheck():
        return 1

    forward, forward_worst, forward_saturations = table("forward", 31)
    inverse, inverse_worst, inverse_saturations = table("inverse", 31)
    margins = worst_margin_per_extent(31)
    print(
        f"Q31 stage tables: worst tie margin forward={float(forward_worst):.6e} "
        f"inverse={float(inverse_worst):.6e} LSB, "
        f"saturations forward={forward_saturations} inverse={inverse_saturations}"
    )
    for extent, margin in margins.items():
        print(f"  extent {extent:3}: worst tie margin {float(margin):.6e} LSB")

    rendered = render(forward, inverse, margins)
    if arguments.write:
        HEADER.write_text(rendered)
        print(f"wrote {HEADER}")
        return 0
    if not HEADER.exists():
        print(f"{HEADER} does not exist; rerun with --write", file=sys.stderr)
        return 1
    if HEADER.read_text() != rendered:
        print(f"{HEADER} is out of date; rerun with --write", file=sys.stderr)
        return 1
    print(f"{HEADER} matches the 50-digit regeneration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
