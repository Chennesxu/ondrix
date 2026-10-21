def acc_dot_wrap(x: buffer[q15], h: buffer[q15]) -> q15:
  return dot(x, h,
             accumulator=exact[40, wrap],
             rounding=nearest_even,
             overflow=saturate)
