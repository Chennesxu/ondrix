def mac(x: buffer[q15], h: buffer[q15]) -> q31:
  acc = 0
  for i in range(len(x)):
    acc = acc + widen(x[i], to=q31) * widen(h[i], to=q31)
  return acc
