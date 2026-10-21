def sl_delay(d: mut buffer[q15], h: buffer[q15], x: q15) -> q15:
  w = d[0:len(h)]
  for k in range(1, len(d)):
    d[len(d) - k] = d[len(d) - k - 1]
  d[0] = x
  return dot(w, h,
             accumulator=exact[40, saturate],
             rounding=nearest_ties_positive,
             overflow=saturate)
