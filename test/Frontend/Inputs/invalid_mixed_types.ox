kernel mixed(lhs: buffer[q15], rhs: buffer[f32]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
