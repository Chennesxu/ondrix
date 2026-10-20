def invalid_literal_scalar_range(x: buffer[q15]) -> q15:
  return x[0] + 32768
