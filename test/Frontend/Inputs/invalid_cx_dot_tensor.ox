def invalid_cx_dot_tensor(lhs: tensor[complex_q15, 8], rhs: tensor[complex_q15, 8]) -> complex_q15:
  return cx_dot(lhs, rhs,
                accumulator=exact[40, saturate],
                rounding=nearest_even,
                overflow=saturate)
