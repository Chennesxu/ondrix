kernel invalid_q31_width(lhs: buffer[q31], rhs: buffer[q31]) -> q31:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
