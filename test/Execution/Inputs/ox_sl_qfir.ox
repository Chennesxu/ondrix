def sl_qfir(x: buffer[q15], h: buffer[q15], y: mut buffer[q15]):
  for n in range(len(y)):
    y[n] = dot(x[n:n + len(h)], h,
               accumulator=exact[40, saturate],
               rounding=nearest_ties_positive,
               overflow=saturate)
