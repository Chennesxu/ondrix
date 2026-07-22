kernel q31_dot(lhs: buffer[q31], rhs: buffer[q31]) -> q31:
  return dot(lhs, rhs,
             accumulator=exact[64, saturate],
             rounding=nearest_even,
             overflow=saturate)
