def invalid_fir_interpolate_shape(
    input: tensor[q15,4], coefficients: tensor[q15,3]) -> tensor[q15,8]:
  return fir_interpolate(input, coefficients, factor=2)
