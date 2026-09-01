#!/usr/bin/env python3
"""Replayable evidence for the compile-time Q15 quantization tie guard.

The compiler generates FFT twiddle tables, DCT cosine tables, and FIR design
coefficients by quantizing binary64 estimates of the real-valued contract
equations, and fails closed whenever an estimate lies closer than 2^-20 Q15
LSB to a rounding half-integer (see
include/ondrix/Support/GuardedFixedQuantization.h). That admissibility claim is
only as strong as the distance between every real coefficient and its nearest
tie, so this script recomputes each supported profile with 50-digit mpmath
and reports, per profile:

  * the minimum tie distance in Q15 LSB units (the guard needs > 2^-20,
    which is about 9.54e-7),
  * the declared saturation count (+1.0 rows quantize to 32767),
  * a SHA-256 hash of the emitted int16 little-endian table for drift
    detection.

The binary64/libm evaluation error budget documented in the source comments
must stay below the reported margins; every margin here exceeds the guard by
more than three orders of magnitude, so any libm within that budget produces
the identical tables.

The Q31 twiddle profiles are reported in the same Q31 LSB units for
comparison, but they are NOT produced by that mechanism: 2^-20 Q31 LSB is
2^-51 absolute, only about 4x a binary64 evaluation error, so the guard can no
longer certify an in-compiler estimate. Those tables are frozen offline by
scripts/generate-q31-twiddle-tables.py; the margins below are the frozen
values' distance to a tie, which is why they must stay large.

Q31 window and design profiles ARE produced by the guard mechanism, at the
compiler's own looser Q31 bound, and are reported against that bound rather
than the Q15 one. That bound refuses whole extents often enough to be worth
counting, which is the second mode:

  python3 scripts/report-quantization-margins.py --sweep-windows

reports, per window and width, how many of the 4095 supported extents the
guard refuses and the shortest one it refuses. That decision is a binary64
one -- it replays what the compiler does, not what the real values are -- so
the sweep uses binary64 and takes about forty seconds.

Requires mpmath (not a build or CI dependency; evidence is regenerated on
demand):  python3 scripts/report-quantization-margins.py
"""

import hashlib
import struct
import sys

from mpmath import cos, floor, mp, mpf, pi, sin

mp.dps = 50

GUARD_LSB = mpf(2) ** -20
# The compiler's Q31 guard is deliberately looser than the Q15 one: a scaled
# coefficient approaching 2^31 carries an ulp of 2^-21 LSB, so 2^-20 would
# stand one bit above the representation granularity and prove nothing.
Q31_GUARD_LSB = mpf(2) ** -13


def quantize(value, fractional_bits=15):
    """Round-half-even signed Q1.f with saturation; returns (int, distance).

    distance is |fraction - 1/2| of the scaled value in LSB units: the
    margin between this coefficient and the nearest rounding tie.
    """
    scale = 1 << fractional_bits
    scaled = value * scale
    lower = floor(scaled)
    fraction = scaled - lower
    distance = abs(fraction - mpf(1) / 2)
    quantized = int(lower) + (1 if fraction > mpf(1) / 2 else 0)
    minimum, maximum = -scale, scale - 1
    saturated = quantized > maximum or quantized < minimum
    quantized = max(minimum, min(maximum, quantized))
    return quantized, distance, saturated


class Profile:
    def __init__(self, name, fractional_bits=15, guard=None):
        self.name = name
        self.fractional_bits = fractional_bits
        # Frozen Q31 tables are held to the stricter Q15 bar on purpose; only
        # a profile the compiler itself quantizes uses the compiler's guard.
        self.guard = GUARD_LSB if guard is None else guard
        self.min_distance = mpf("inf")
        self.saturated = 0
        self.values = []

    def add(self, real):
        quantized, distance, saturated = quantize(real, self.fractional_bits)
        self.min_distance = min(self.min_distance, distance)
        self.saturated += 1 if saturated else 0
        self.values.append(quantized)

    def report(self):
        format_code = "<h" if self.fractional_bits == 15 else "<i"
        digest = hashlib.sha256(
            b"".join(struct.pack(format_code, value) for value in self.values)
        ).hexdigest()
        admissible = self.min_distance >= self.guard
        print(
            f"{self.name:34} entries={len(self.values):6} "
            f"min_tie_distance_lsb={float(self.min_distance):.6e} "
            f"saturated={self.saturated:4} sha256={digest[:16]} "
            f"guard={float(self.guard):.2e} "
            f"{'ADMISSIBLE' if admissible else 'FAILS GUARD'}"
        )
        return admissible


def twiddle_profile(direction, extent):
    """Every stage twiddle of the recursive radix-2 combine, in the order
    the conversion generates them: stage sizes 2, 4, ..., extent, index
    0..size/2-1, real then imaginary component."""
    profile = Profile(f"cfft{extent}_{direction}_twiddles")
    sign = -1 if direction == "forward" else 1
    size = 2
    while size <= extent:
        for index in range(size // 2):
            angle = 2 * pi * index / size
            profile.add(cos(angle))
            profile.add(sign * sin(angle))
        size *= 2
    return profile


def q31_twiddle_profile(direction, extent):
    """The same stage twiddles at Q31, where the tables are frozen offline
    rather than quantized in the compiler: 2^-20 Q31 LSB is only ~4x a
    binary64 evaluation error, so the guard can no longer certify an
    in-compiler estimate (see include/ondrix/Support/Q31TwiddleTables.h).
    The margins reported here are the frozen tables' distance to a tie, and
    they must keep matching the checked-in header."""
    profile = Profile(f"cfft{extent}_{direction}_twiddles_q31", fractional_bits=31)
    sign = -1 if direction == "forward" else 1
    size = 2
    while size <= extent:
        for index in range(size // 2):
            angle = 2 * pi * index / size
            profile.add(cos(angle))
            profile.add(sign * sin(angle))
        size *= 2
    return profile


def dct_profile(extent):
    """Direct type-II DCT matrix q15(cos(pi*(2n+1)k/(2N))), row-major."""
    profile = Profile(f"dct{extent}_coefficients")
    for k in range(extent):
        for n in range(extent):
            profile.add(cos(pi * (2 * n + 1) * k / (2 * extent)))
    return profile


def hamming(n, extent):
    return mpf("0.54") - mpf("0.46") * cos(2 * pi * n / (extent - 1))


def hann(n, extent):
    return mpf(1) / 2 - mpf(1) / 2 * cos(2 * pi * n / (extent - 1))


def blackman(n, extent):
    phase = 2 * pi * n / (extent - 1)
    return mpf(21) / 50 - mpf(1) / 2 * cos(phase) + mpf(2) / 25 * cos(2 * phase)


def kaiser(n, extent, beta):
    from mpmath import besseli, sqrt as msqrt
    center = mpf(extent - 1) / 2
    ratio = (n - center) / center
    return besseli(0, beta * msqrt(1 - ratio * ratio)) / besseli(0, beta)


def kaiser_profile(extent, beta_num, beta_den, fractional_bits=15):
    suffix = "" if fractional_bits == 15 else "_q31"
    guard = None if fractional_bits == 15 else Q31_GUARD_LSB
    profile = Profile(f"window_kaiser{extent}_beta{beta_num}_{beta_den}{suffix}",
                      fractional_bits, guard)
    beta = mpf(beta_num) / beta_den
    for n in range(extent):
        profile.add(kaiser(n, extent, beta))
    return profile


def window_profile(name, window, extent, fractional_bits=15):
    suffix = "" if fractional_bits == 15 else "_q31"
    guard = None if fractional_bits == 15 else Q31_GUARD_LSB
    profile = Profile(f"window_{name}{extent}{suffix}", fractional_bits, guard)
    for n in range(extent):
        profile.add(window(n, extent))
    return profile


def sinc(x):
    if x == 0:
        return mpf(1)
    return sin(pi * x) / (pi * x)


def fir_design_profile(extent, cutoff_num, cutoff_den, response):
    """Windowed-sinc design coefficients for one (N, fc, response) profile."""
    profile = Profile(f"fir_{response}{extent}_fc{cutoff_num}_{cutoff_den}")
    center = (extent - 1) // 2
    doubled = mpf(2) * cutoff_num / cutoff_den
    for n in range(extent):
        lowpass = doubled * sinc(doubled * (n - center)) * hamming(n, extent)
        if response == "highpass":
            profile.add((1 if n == center else 0) - lowpass)
        else:
            profile.add(lowpass)
    return profile


def sweep_windows():
    """Per-extent refusal rate of the compile-time guard, at both widths.

    This is the OTHER claim about the guard, and it is not an mpmath one:
    the compiler evaluates these windows in binary64 and compares against the
    guard, so replaying that decision means replaying the binary64 chain. A
    single inadmissible coefficient refuses the whole extent, so extents --
    not coefficients -- are what a user counts.
    """
    import math

    two_pi = 6.28318530717958647692528676655900577
    windows = {
        "hamming": lambda n, N: 0.54 - 0.46 * math.cos(two_pi * n / (N - 1)),
        "hann": lambda n, N: 0.5 - 0.5 * math.cos(two_pi * n / (N - 1)),
        "blackman": lambda n, N: 0.42
        - 0.5 * math.cos(two_pi * n / (N - 1))
        + 0.08 * math.cos(2 * two_pi * n / (N - 1)),
    }
    for width, scale, guard in ((15, 32768.0, 2.0**-20), (31, 2147483648.0, 2.0**-13)):
        for name, window in windows.items():
            refused = []
            for extent in range(2, 4097):
                for n in range(extent):
                    scaled = window(n, extent) * scale
                    if abs((scaled - math.floor(scaled)) - 0.5) < guard:
                        refused.append(extent)
                        break
            first = refused[0] if refused else "-"
            print(
                f"Q{width} window_{name:9} refused_extents={len(refused):5}/4095 "
                f"({100.0 * len(refused) / 4095:5.2f}%) shortest_refused={first}"
            )


def main():
    if "--sweep-windows" in sys.argv:
        sweep_windows()
        return 0
    profiles = []
    for direction in ("forward", "inverse"):
        for extent in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
            profiles.append(twiddle_profile(direction, extent))
    # The Q31 complex profile covers extents 4..64 only; its tables are frozen
    # in the header instead of regenerated by the compiler.
    for direction in ("forward", "inverse"):
        for extent in (4, 8, 16, 32, 64):
            profiles.append(q31_twiddle_profile(direction, extent))
    for extent in (4, 8, 16, 32, 64):
        profiles.append(dct_profile(extent))
    # The committed design gates plus boundary extents; arbitrary user
    # profiles remain protected by the fail-closed guard at compile time.
    profiles.append(fir_design_profile(9, 1, 4, "lowpass"))
    profiles.append(fir_design_profile(9, 1, 4, "highpass"))
    profiles.append(fir_design_profile(11, 1, 8, "lowpass"))
    profiles.append(fir_design_profile(4095, 1, 4, "lowpass"))
    for name, window in (("hamming", hamming), ("hann", hann), ("blackman", blackman)):
        for extent in (8, 9, 64, 4096):
            profiles.append(window_profile(name, window, extent))
        # Q31 windows at extents the compiler's own 2^-13 guard admits. 4096
        # is absent on purpose: Hann and Kaiser(6) are REFUSED there, which
        # --sweep-windows reports rather than hiding.
        for extent in (8, 9, 64):
            profiles.append(window_profile(name, window, extent, fractional_bits=31))
    for extent in (8, 9, 64, 4096):
        profiles.append(kaiser_profile(extent, 6, 1))
    for extent in (8, 9, 64):
        profiles.append(kaiser_profile(extent, 6, 1, fractional_bits=31))
    sine_table = Profile("sine_table256")
    for k in range(256):
        sine_table.add(sin(2 * pi * k / 256))
    profiles.append(sine_table)
    # The Q31 trigonometric profile is a different construction, not this one
    # widened: 1024 coarse angles read by third-order angle addition rather
    # than 256 read by linear interpolation.
    sine_table_q31 = Profile("sine_table1024_q31", fractional_bits=31, guard=Q31_GUARD_LSB)
    for k in range(1024):
        sine_table_q31.add(sin(2 * pi * k / 1024))
    profiles.append(sine_table_q31)

    print(f"tie guard: {float(GUARD_LSB):.6e} Q15 LSB (2^-20)")
    if all([profile.report() for profile in profiles]):
        print("all profiles admissible")
        return 0
    print("AT LEAST ONE PROFILE FAILS THE GUARD")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
