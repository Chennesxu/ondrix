def sl_taps(x: buffer[q15], h: buffer[q15]) -> q15:
  t = zeros(q15, 8)
  for k in range(len(x)):
    t[k] = x[k]
  return dot(t, h[0:8],
             accumulator=exact[40, saturate],
             rounding=nearest_ties_positive,
             overflow=saturate)
