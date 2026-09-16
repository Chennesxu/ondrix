def q31_dot_raw_high(lhs: buffer[q31], rhs: buffer[q31]) -> q31:
  return dot(lhs, rhs,
             product=raw_high,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
