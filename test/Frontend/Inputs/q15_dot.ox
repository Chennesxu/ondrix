def q15_dot(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
