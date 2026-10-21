def acc_dot(x: buffer[q15], h: buffer[q15]) -> q15:
  return dot(x, h,
             accumulator=exact[40, saturate],
             rounding=nearest_ties_positive,
             overflow=saturate)
