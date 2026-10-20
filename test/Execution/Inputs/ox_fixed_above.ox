def above(x: buffer[q15], t: q15) -> bool:
  return x[0] > t and -x[1] <= 100
