def invalid_cx_dot_conjugate(lhs: buffer[complex_q15, 8], rhs: buffer[complex_q15, 8]) -> complex_q15:
  return cx_dot(lhs, rhs, conjugate=maybe,
                accumulator=exact[40, saturate],
                rounding=nearest_even,
                overflow=saturate)
