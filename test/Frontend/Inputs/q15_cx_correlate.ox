def q15_cx_correlate(lhs: buffer[complex_q15, 64], rhs: buffer[complex_q15, 64]) -> complex_q15:
  return cx_dot(lhs, rhs, conjugate=true,
                accumulator=exact[40, saturate],
                rounding=nearest_even,
                overflow=saturate)
