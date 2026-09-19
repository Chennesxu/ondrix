def invalid_cx_dot_accumulator(lhs: buffer[complex_q15, 8], rhs: buffer[complex_q15, 8]) -> complex_q15:
  return cx_dot(lhs, rhs,
                accumulator=exact[64, saturate],
                rounding=nearest_even,
                overflow=saturate)
