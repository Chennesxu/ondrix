def q15_dot_ties_positive(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_ties_positive,
             overflow=saturate)
