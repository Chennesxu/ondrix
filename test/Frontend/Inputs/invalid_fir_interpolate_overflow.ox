def invalid_fir_interpolate_overflow(
    input: tensor[q15,9223372036854775807],
    coefficients: tensor[q15,2]) -> tensor[q15,1]:
  return fir_interpolate(input, coefficients, factor=2)
