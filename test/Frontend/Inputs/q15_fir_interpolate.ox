def q15_fir_interpolate(
    input: tensor[q15,4], coefficients: tensor[q15,3]) -> tensor[q15,9]:
  return fir_interpolate(input, coefficients, factor=2)
