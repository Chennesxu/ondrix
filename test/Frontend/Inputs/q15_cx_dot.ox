def q15_cx_dot(lhs: buffer[complex_q15, 64], rhs: buffer[complex_q15, 64]) -> complex_q15:
  return cx_dot(lhs, rhs,
                accumulator=exact[32, saturate],
                rounding=nearest_ties_positive,
                overflow=saturate)
