def invalid_cx_dot_q31_width(lhs: buffer[complex_q31, 8], rhs: buffer[complex_q31, 8]) -> complex_q31:
  return cx_dot(lhs, rhs,
                accumulator=exact[40, saturate],
                rounding=nearest_even,
                overflow=saturate)
