#!/usr/bin/env python3
"""Regenerate the frozen f32 arctangent-turn polynomial.

The interleaved f32 `ondrix.cx_phase` profile folds every input into an
octant, so its arctangent argument is a ratio in [0, 1] and one polynomial
covers the whole reduced domain -- no range reduction, no table, no second
division. The coefficients approximate

    atan(r) / (2*pi) = r * g(r*r),  g of degree 8 in u = r*r

by a Remez exchange under RELATIVE error, computed here at 60 decimal digits
and rounded once to binary32. They are frozen rather than evaluated in the
compiler for a reason the Q31 twiddles do not share: a fit is not a
closed-form call, so there is no libm function the build could evaluate. What
pins them instead is this script plus the emitted-bit test.

Nine terms is the smallest count that lands under the format: eight reach
9.9e-8 relative, about 1.7 binary32 epsilons, and nine reach 1.5e-8.

Usage (requires mpmath; not a build or CI dependency):

  python3 scripts/generate-f32-arctangent-turn.py            # verify only
  python3 scripts/generate-f32-arctangent-turn.py --write    # rewrite header
"""

import argparse
import pathlib
import struct
import sys

import mpmath as mp

HEADER = pathlib.Path(__file__).resolve().parent.parent / "include/ondrix/Support/F32ArctangentTurn.h"
DEGREE = 8
GRID = 20000


def turn_over_root(u):
    """atan(sqrt(u)) / (2*pi*sqrt(u)), the even factor the odd fit reads."""
    if u == 0:
        return 1 / (2 * mp.pi)
    root = mp.sqrt(u)
    return mp.atan(root) / (2 * mp.pi * root)


def remez(degree, iterations=60):
    count = degree + 2
    nodes = [mp.mpf(1) / 2 - mp.cos(mp.pi * k / (count - 1)) / 2 for k in range(count)]
    coefficients = None
    for _ in range(iterations):
        matrix = mp.matrix(count, count)
        rhs = mp.matrix(count, 1)
        for row, node in enumerate(nodes):
            value = turn_over_root(node)
            for column in range(degree + 1):
                matrix[row, column] = node**column
            matrix[row, degree + 1] = (-1) ** row * value
            rhs[row] = value
        solution = mp.lu_solve(matrix, rhs)
        coefficients = [solution[i] for i in range(degree + 1)]

        def relative_error(x):
            fit = mp.polyval(list(reversed(coefficients)), x)
            exact = turn_over_root(x)
            return (fit - exact) / exact

        samples = [mp.mpf(k) / GRID for k in range(GRID + 1)]
        errors = [relative_error(x) for x in samples]
        extrema = [samples[0]] if abs(errors[0]) > abs(errors[1]) else []
        for k in range(1, len(samples) - 1):
            if (errors[k] - errors[k - 1]) * (errors[k + 1] - errors[k]) <= 0:
                extrema.append(samples[k])
        if abs(errors[-1]) > abs(errors[-2]):
            extrema.append(samples[-1])
        if len(extrema) < count:
            break
        nodes = extrema[:count]
    peak = max(abs(relative_error(mp.mpf(k) / GRID)) for k in range(GRID + 1))
    return coefficients, peak


def to_float32(value):
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def render(coefficients):
    return [float.hex(to_float32(c)).replace("0x1.", "0x1.").rstrip() + "f" for c in coefficients]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()

    mp.mp.dps = 60
    coefficients, peak = remez(DEGREE)
    print(f"{DEGREE + 1} terms, max relative error {mp.nstr(peak, 5)}")
    literals = []
    for index, coefficient in enumerate(coefficients):
        rounded = to_float32(coefficient)
        literal = float.hex(rounded)
        # C hex float literals keep the significand the header spells.
        mantissa, exponent = literal.split("p")
        mantissa = mantissa.rstrip("0").rstrip(".")
        literals.append(f"{mantissa}p{exponent}f")
        print(f"  u^{index}: {mp.nstr(coefficient, 20)} -> {literals[-1]}")

    text = HEADER.read_text()
    frozen = [token.strip().rstrip(",") for token in
              text.split("kCoefficients[] = {")[1].split("};")[0].replace("\n", " ").split(",")
              if token.strip()]
    if frozen == literals:
        print(f"{HEADER.name}: frozen coefficients match")
        return 0
    print(f"{HEADER.name}: frozen coefficients DIFFER", file=sys.stderr)
    for was, now in zip(frozen, literals):
        if was != now:
            print(f"  {was} -> {now}", file=sys.stderr)
    if not arguments.write:
        return 1
    body = ",\n      ".join(", ".join(literals[i:i + 3]) for i in range(0, len(literals), 3))
    head, rest = text.split("kCoefficients[] = {")
    _, tail = rest.split("};", 1)
    HEADER.write_text(f"{head}kCoefficients[] = {{\n      {body}}};{tail}")
    print(f"{HEADER.name}: rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
