def invalid_fir_interpolate_factor(
    input: tensor[q15,4], coefficients: tensor[q15,3]) -> tensor[q15,12]:
  return fir_interpolate(input, coefficients, factor=3)
