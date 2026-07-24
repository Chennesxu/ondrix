def invalid_constexpr_lhs(
    lhs: constexpr[q15] = [1, 2, 3], rhs: buffer[q15, 3]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
