def gap(
    x: buffer[q15, 8],
    c: constexpr[q15] = [1000, -2000, 3000, -4000, 5000, -6000, 7000, -8000]) -> q15:
  return dot(x, c) - x[0]
