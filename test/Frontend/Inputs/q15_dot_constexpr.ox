kernel q15_dot_constexpr(
    lhs: buffer[q15, 8], rhs: constexpr[q15] = [1, -2, 3, -4, 5, -6, 7, -8]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
