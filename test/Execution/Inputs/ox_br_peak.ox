def br_peak(x: buffer[q15], y: mut buffer[q15]) -> i64:
  top = x[0] if len(x) > 0 else -32768
  at = -1
  k = 0
  for i in range(len(x)):
    if x[i] > top or at < 0:
      top = x[i]
      at = k
    y[i] = top
    k = k + 1
  return at
