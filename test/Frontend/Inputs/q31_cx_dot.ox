def q31_cx_dot(lhs: buffer[complex_q31, 64], rhs: buffer[complex_q31, 64]) -> complex_q31:
  return cx_dot(lhs, rhs, conjugate=true,
                accumulator=exact[64, saturate],
                rounding=nearest_even,
                overflow=saturate)
