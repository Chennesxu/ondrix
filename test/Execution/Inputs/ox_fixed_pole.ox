def pole(x: buffer[q15]) -> q15:
  y = 0
  for i in range(len(x)):
    y = y * 29491 + x[i] * 3277
  return y
