def bad_operand(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, coefficients,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
