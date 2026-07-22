kernel bad_width(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[39, saturate],
             rounding=nearest_even,
             overflow=saturate)
